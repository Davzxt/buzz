"""Sparse prototype classifier: fixed encoder + additive prototype memory."""
import numpy as np

from .encoder import SparseExpansionEncoder
from .memory import PrototypeMemory


class SparsePrototypeClassifier:
    """Learns in one pass without gradients. Each `fit_task` call is a single update."""

    def __init__(self, input_dim, expansion_dim=2000, fan_in=8, k_active=64, seed=0):
        self.encoder = SparseExpansionEncoder(
            input_dim, expansion_dim=expansion_dim, fan_in=fan_in, k_active=k_active, seed=seed
        )
        self.memory = PrototypeMemory(expansion_dim)

    def fit_task(self, X, y):
        y = np.asarray(y)
        codes = self.encoder.encode(X)
        for label in np.unique(y):
            self.memory.learn(codes[y == label], label)

    def predict(self, X):
        return self.memory.predict(self.encoder.encode(X))

    def memory_floats(self):
        # Prototype rows plus the encoder's projection, which is also stored state.
        return self.memory.memory_floats() + int(self.encoder.projection.size)
