"""Character-level data: vocabulary, encoding and sliding context windows."""
from pathlib import Path

import numpy as np


class CharVocab:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return np.array([self.index[c] for c in text], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.chars[int(i)] for i in ids)


def read_corpus(path):
    return Path(path).read_text(encoding="utf-8")


def make_windows(ids, context):
    """Return (contexts, targets). Row i predicts ids[i + context] from the previous `context` ids."""
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(ids), context + 1)
    return np.ascontiguousarray(view[:, :context]), np.ascontiguousarray(view[:, context])
