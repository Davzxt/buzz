"""Treino reproduzível do componente de decisão. Roda em segundos na CPU (local, Colab ou VPS).

    python -m buzz.train --out models/decision.joblib --report docs/train_report.json
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from .data import LABELS, splits
from .decision import Decision


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    acc, e = (pred == y), 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def evaluate(dec, test):
    p = dec.proba([r["text"] for r in test])
    classes = list(dec.clf.classes_)
    y = np.array([classes.index(r["label"]) for r in test])
    return {"acc": float((p.argmax(1) == y).mean()), "ece": ece(p, y), "temp": dec.temp,
            "acc_conf>=0.6": float((p.argmax(1) == y)[p.max(1) >= 0.6].mean()) if (p.max(1) >= 0.6).any() else None,
            "coverage@0.6": float((p.max(1) >= 0.6).mean()), "n_test": len(test)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/decision.joblib")
    ap.add_argument("--report", default="docs/train_report.json")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train, val, test = splits(a.n_train, seed=a.seed)
    t0 = time.time()
    dec = Decision().fit(train, val)
    secs = time.time() - t0
    rep = {"labels": LABELS, "train_seconds": round(secs, 2), "n_train": len(train), "n_val": len(val),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 1),
           "test_unseen_templates": evaluate(dec, test),
           "note": "dados sintéticos; templates de teste disjuntos dos de treino/validação"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    dec.save(a.out)
    Path(a.report).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
