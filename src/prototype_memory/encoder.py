"""Random sparse expansion with k-winner-take-all (FlyHash-style encoder)."""
import numpy as np


class SparseExpansionEncoder:
    """Map dense inputs to sparse binary codes.

    Every expansion unit samples `fan_in` input dimensions once, at construction.
    The projection is fixed and never trained. For each input, the `k_active`
    units with the largest response are set to 1 and the rest to 0.
    """

    def __init__(self, input_dim, expansion_dim=2000, fan_in=8, k_active=64, seed=0):
        if not 0 < fan_in <= input_dim:
            raise ValueError("fan_in must be in (0, input_dim]")
        if not 0 < k_active <= expansion_dim:
            raise ValueError("k_active must be in (0, expansion_dim]")
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.expansion_dim = expansion_dim
        self.fan_in = fan_in
        self.k_active = k_active
        self.projection = np.zeros((expansion_dim, input_dim), dtype=np.float32)
        for unit in range(expansion_dim):
            self.projection[unit, rng.choice(input_dim, size=fan_in, replace=False)] = 1.0

    def encode(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"expected shape (n, {self.input_dim}), got {X.shape}")
        response = X @ self.projection.T
        winners = np.argpartition(-response, self.k_active - 1, axis=1)[:, : self.k_active]
        codes = np.zeros(response.shape, dtype=np.float32)
        np.put_along_axis(codes, winners, 1.0, axis=1)
        return codes
