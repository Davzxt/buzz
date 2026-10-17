"""Additive per-class prototype memory over sparse codes."""
import numpy as np


class PrototypeMemory:
    """One row per class: the running sum of the codes seen for that class.

    Rows are independent. Learning a class touches only its own row, so adding
    classes or updating one class cannot change the stored prototypes of others.
    """

    def __init__(self, code_dim):
        self.code_dim = code_dim
        self.labels = []
        self._index = {}
        self._sums = np.zeros((0, code_dim), dtype=np.float64)

    def add_class(self, label):
        label = int(label)
        if label in self._index:
            return
        self._index[label] = len(self.labels)
        self.labels.append(label)
        self._sums = np.vstack([self._sums, np.zeros((1, self.code_dim))])

    def learn(self, codes, label):
        self.add_class(label)
        self._sums[self._index[int(label)]] += np.asarray(codes, dtype=np.float64).sum(axis=0)

    def prototypes(self):
        norms = np.linalg.norm(self._sums, axis=1, keepdims=True)
        return self._sums / np.maximum(norms, 1e-12)

    def scores(self, codes):
        # Every code has the same number of active bits, so this dot product is a
        # cosine similarity up to a constant factor that does not change the argmax.
        return np.asarray(codes, dtype=np.float64) @ self.prototypes().T

    def predict(self, codes):
        if not self.labels:
            raise RuntimeError("memory is empty; learn at least one class first")
        return np.asarray(self.labels)[np.argmax(self.scores(codes), axis=1)]

    def snapshot(self):
        return self._sums.copy()

    def memory_floats(self):
        return int(self._sums.size)
