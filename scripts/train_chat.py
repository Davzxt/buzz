"""Train the dense character model on the pt-BR corpus, with checkpoints and a validation report.

    python scripts/train_chat.py --corpus data/ptbr_mixed_corpus.txt --out checkpoints

Writes checkpoints/last.npz after every epoch, checkpoints/best.npz when validation improves,
and checkpoints/train_report.json with the config and the validation bits per character,
compared with a Witten-Bell n-gram on the same validation windows.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from charlm import Adam, CharVocab, DenseCharMLP, WittenBellNGram, make_windows, read_corpus  # noqa: E402
from charlm.checkpoint import save_checkpoint  # noqa: E402

LN2 = np.log(2)


def split_blocks(ids, block, every):
    """Every `every`-th block of `block` ids goes to validation. The corpus is ordered by source,
    so a contiguous tail split would validate on one source only."""
    parts = [ids[i : i + block] for i in range(0, len(ids), block)]
    val = [b for k, b in enumerate(parts) if k % every == every - 1]
    train = [b for k, b in enumerate(parts) if k % every != every - 1]
    return train, val


def windows_of(parts, context):
    pairs = [make_windows(part, context) for part in parts if len(part) > context]
    return np.concatenate([c for c, _ in pairs]), np.concatenate([y for _, y in pairs])


def bits_per_char(model, ctx, y):
    return float(-model.log_prob(ctx, y).mean() / LN2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/ptbr_mixed_corpus.txt")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--context", type=int, default=16)
    p.add_argument("--emb", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--eval_windows", type=int, default=50_000)
    p.add_argument("--ngram_order", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = read_corpus(args.corpus)
    vocab = CharVocab(text)
    ids = vocab.encode(text)
    C = args.context
    train_parts, val_parts = split_blocks(ids, block=2000, every=20)
    train_ids = np.concatenate(train_parts)
    ctx_tr, y_tr = windows_of(train_parts, C)
    ctx_va, y_va = windows_of(val_parts, C)
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(y_va), size=min(args.eval_windows, len(y_va)), replace=False)
    ctx_va, y_va = ctx_va[sub], y_va[sub]

    config = {k: getattr(args, k) for k in ["context", "emb", "hidden", "epochs", "batch", "lr", "seed"]}
    config["vocab_size"] = len(vocab)
    config["train_windows"] = int(len(y_tr))
    config["val_windows"] = int(len(y_va))

    ngram = WittenBellNGram(args.ngram_order, len(vocab)).fit(train_ids)
    ngram_bpc = float(-ngram.log_prob(ctx_va, y_va).mean() / LN2)
    print(f"n-gram (order {args.ngram_order}) validation bpc: {ngram_bpc:.3f}", flush=True)

    model = DenseCharMLP(len(vocab), C, args.emb, args.hidden, seed=args.seed)
    opt = Adam(model.params, args.lr)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        order = rng.permutation(len(y_tr))
        losses = []
        for s in range(0, len(order), args.batch):
            b = order[s : s + args.batch]
            loss, grads = model.loss_and_grads(ctx_tr[b], y_tr[b])
            opt.step(grads)
            losses.append(loss)
        val_bpc = bits_per_char(model, ctx_va, y_va)
        record = {"epoch": epoch, "train_loss_nats": float(np.mean(losses)),
                  "val_bpc": val_bpc, "seconds": round(time.perf_counter() - start, 1)}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        save_checkpoint(out / "last.npz", model, vocab, config)
        if val_bpc < best:
            best = val_bpc
            save_checkpoint(out / "best.npz", model, vocab, config)

    report = {"config": config, "ngram_val_bpc": ngram_bpc, "best_val_bpc": best,
              "beats_ngram": best < ngram_bpc, "history": history}
    (out / "train_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best validation bpc {best:.3f} vs n-gram {ngram_bpc:.3f}")


if __name__ == "__main__":
    main()
