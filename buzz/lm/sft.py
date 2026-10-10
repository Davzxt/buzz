"""Etapa 2 (ainda do SEU modelo): ensina o formato de conversa. Perda só na resposta.
Entrada: jsonl com {"prompt": "...", "response": "..."} (seus dados, ou datasets abertos em PT-BR com licença compatível).

    python -m lm.sft --run runs/buzz-lm --data data/chat.jsonl --epochs 3 --out runs/buzz-chat
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import GPT, Cfg, cfg_dict, load


def encode(tok, p, r, ctx):
    u, b, e = tok.token_to_id("<usuario>"), tok.token_to_id("<buzz>"), tok.token_to_id("<eos>")
    pi = [u] + tok.encode(p).ids + [b]
    ri = tok.encode(r).ids + [e]
    ids = (pi + ri)[:ctx + 1]
    lab = ([-100] * len(pi) + ri)[:ctx + 1]
    return ids, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/buzz-lm")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/buzz-chat")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(f"{a.run}/tokenizer.json")
    m = load(f"{a.run}/best.pt", dev).train()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    random.Random(0).shuffle(rows)
    cut = max(1, len(rows) // 20)
    val, tr = rows[:cut], rows[cut:]
    ex = lambda rs: [encode(tok, r["prompt"], r["response"], m.c.ctx) for r in rs]
    tr, val = ex(tr), ex(val)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)

    def run(batch_rows):
        L = max(len(i) for i, _ in batch_rows)
        x = torch.tensor([i[:-1] + [0] * (L - len(i)) for i, _ in batch_rows], device=dev)
        y = torch.tensor([l[1:] + [-100] * (L - len(l)) for _, l in batch_rows], device=dev)
        return m(x, y)[1]

    t0 = time.time()
    for ep in range(a.epochs):
        random.shuffle(tr)
        for i in range(0, len(tr), a.batch):
            loss = run(tr[i:i + a.batch])
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
        m.eval()
        with torch.no_grad():
            v = sum(run(val[i:i + a.batch]).item() for i in range(0, len(val), a.batch)) / max(1, len(val) // a.batch)
        m.train()
        print(json.dumps({"epoch": ep + 1, "train_loss": round(loss.item(), 3), "val_loss": round(v, 3),
                          "min": round((time.time() - t0) / 60, 1)}), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{a.run}/tokenizer.json", out / "tokenizer.json")
    torch.save({"model": m.state_dict(), "cfg": cfg_dict(m.c)}, out / "best.pt")


if __name__ == "__main__":
    main()
