"""Pré-treino do zero com ORÇAMENTO DE TEMPO (o cosseno de LR é função do tempo decorrido).

    python -m lm.train --data data/lm --minutes 180 --out runs/buzz-lm      # Colab T4: ~3 h
    python -m lm.train --data data/lm_smoke --minutes 2 --dim 128 --layers 4 --heads 4 --hidden 352 --ctx 128 --batch 16  # smoke
Retoma sozinho se runs/.../last.pt existir (útil quando o Colab desconecta).
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .model import GPT, Cfg, cfg_dict


def batch(data, bs, ctx, dev):
    ix = np.random.randint(0, len(data) - ctx - 1, bs)
    x = torch.from_numpy(np.stack([data[i:i + ctx] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + ctx] for i in ix]).astype(np.int64))
    return x.to(dev), y.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lm")
    ap.add_argument("--out", default="runs/buzz-lm")
    ap.add_argument("--minutes", type=float, default=180)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    for k, v in vars(Cfg()).items():
        ap.add_argument(f"--{k}", type=int, default=v)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(a.data) / "tokenizer.json", out / "tokenizer.json")
    tr = np.memmap(Path(a.data) / "train.bin", dtype=np.uint16, mode="r")
    va = np.memmap(Path(a.data) / "val.bin", dtype=np.uint16, mode="r")
    from tokenizers import Tokenizer
    cfg = Cfg(vocab=Tokenizer.from_file(str(out / "tokenizer.json")).get_vocab_size(), dim=a.dim, layers=a.layers,
              heads=a.heads, ctx=a.ctx, hidden=a.hidden)
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    use_amp = dev == "cuda"
    adt = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and adt == torch.float16)
    it, spent = 0, 0.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); it, spent = ck["it"], ck["spent"]
        print(f"retomando it={it} gasto={spent / 60:.1f}min")
    print(f"params={model.n_params() / 1e6:.1f}M device={dev} tokens_treino={len(tr):,} (~{len(tr) * 1e0 / 1e6:.0f}M)")

    @torch.no_grad()
    def val_loss():
        model.eval()
        ls = []
        for _ in range(20):
            x, y = batch(va, a.batch, cfg.ctx, dev)
            with torch.autocast(dev, dtype=adt, enabled=use_amp):
                ls.append(model(x, y)[1].item())
        model.train()
        return sum(ls) / len(ls)

    def save(name, best=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg_dict(cfg), "it": it,
                    "spent": spent, "best": best}, out / name)

    total, best, t_start, model_t = a.minutes * 60, float("inf"), time.time(), spent
    model.train()
    while spent < total:
        frac = spent / total
        lr = a.lr * min(1.0, (it + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(a.accum):
            x, y = batch(tr, a.batch, cfg.ctx, dev)
            with torch.autocast(dev, dtype=adt, enabled=use_amp):
                loss = model(x, y)[1] / a.accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        it += 1
        spent = model_t + time.time() - t_start
        if not math.isfinite(loss.item()):
            raise SystemExit("loss não-finita: abortando (critério de parada)")
        if it % 25 == 0:
            print(json.dumps({"it": it, "loss": round(loss.item() * a.accum, 3), "lr": round(lr, 6), "min": round(spent / 60, 1)}), flush=True)
        if it % a.eval_every == 0:
            v = val_loss()
            print(json.dumps({"it": it, "val_loss": round(v, 3), "ppl": round(math.exp(v), 1)}), flush=True)
            save("last.pt")
            if v < best:
                best = v; save("best.pt", best)
    v = val_loss(); save("last.pt")
    if v < best:
        save("best.pt", v); best = v
    print(json.dumps({"final_val_loss": round(best, 3), "ppl": round(math.exp(best), 1), "iters": it, "minutes": round(spent / 60, 1)}))


if __name__ == "__main__":
    main()
