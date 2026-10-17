"""Witten-Bell interpolated character n-gram. A strong, fast, non-neural reference."""
import numpy as np


class WittenBellNGram:
    def __init__(self, order, vocab_size):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.vocab_size = vocab_size
        self.total = [dict() for _ in range(order)]
        self.types = [dict() for _ in range(order)]
        self.count = [dict() for _ in range(order)]

    def fit(self, ids):
        seq = np.asarray(ids).tolist()
        for L in range(self.order):
            total, types, count = self.total[L], self.types[L], self.count[L]
            for i in range(L, len(seq)):
                ctx = tuple(seq[i - L : i])
                key = ctx + (seq[i],)
                total[ctx] = total.get(ctx, 0) + 1
                if key in count:
                    count[key] += 1
                else:
                    count[key] = 1
                    types[ctx] = types.get(ctx, 0) + 1
        return self

    def log_prob(self, ctx, y):
        """ctx: (n, C) array with C >= order - 1; y: (n,) targets. Returns natural-log probabilities."""
        ctx = np.asarray(ctx)
        y = np.asarray(y)
        C = ctx.shape[1]
        if C < self.order - 1:
            raise ValueError("context length must be at least order - 1")
        out = np.empty(len(y), dtype=np.float64)
        rows = ctx.tolist()
        for n, (row, target) in enumerate(zip(rows, y.tolist())):
            p = 1.0 / self.vocab_size
            for L in range(self.order):
                h = tuple(row[C - L :]) if L else ()
                tot = self.total[L].get(h, 0)
                if tot == 0:
                    break
                t = self.types[L][h]
                c = self.count[L].get(h + (target,), 0)
                p = (c + t * p) / (tot + t)
            out[n] = np.log(p)
        return out

    def memory_floats(self):
        return int(sum(len(d) for d in self.count))
