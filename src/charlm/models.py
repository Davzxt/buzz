"""Character language models: dense MLP, sparse-code readout, and sparse prototype memory."""
import numpy as np

from .optim import Adam, log_softmax


class DenseCharMLP:
    """Context embeddings -> tanh hidden layer -> softmax over characters. Trained by backprop."""

    def __init__(self, vocab_size, context, emb=32, hidden=256, seed=0):
        rng = np.random.default_rng(seed)
        self.context, self.emb = context, emb
        d = context * emb
        self.params = {
            "E": rng.normal(0, 0.1, (vocab_size, emb)).astype(np.float32),
            "W1": rng.normal(0, 1 / np.sqrt(d), (d, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, np.float32),
            "W2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, vocab_size)).astype(np.float32),
            "b2": np.zeros(vocab_size, np.float32),
        }

    def _forward(self, ctx):
        p = self.params
        x = p["E"][ctx].reshape(len(ctx), -1)
        h = np.tanh(x @ p["W1"] + p["b1"])
        return x, h, h @ p["W2"] + p["b2"]

    def loss_and_grads(self, ctx, y):
        p = self.params
        B = len(y)
        x, h, z = self._forward(ctx)
        lp = log_softmax(z)
        loss = -lp[np.arange(B), y].mean()
        dz = np.exp(lp)
        dz[np.arange(B), y] -= 1
        dz /= B
        da = (dz @ p["W2"].T) * (1 - h * h)
        dx = (da @ p["W1"].T).reshape(B, self.context, self.emb)
        dE = np.zeros_like(p["E"])
        np.add.at(dE, ctx, dx)
        grads = {"W2": h.T @ dz, "b2": dz.sum(0), "W1": x.T @ da, "b1": da.sum(0), "E": dE}
        return float(loss), grads

    def fit(self, ctx, y, epochs, batch_size, lr, seed=0):
        opt = Adam(self.params, lr)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                b = order[start : start + batch_size]
                _, grads = self.loss_and_grads(ctx[b], y[b])
                opt.step(grads)

    def log_prob(self, ctx, y, batch=65536):
        out = []
        for start in range(0, len(y), batch):
            _, _, z = self._forward(ctx[start : start + batch])
            lp = log_softmax(z)
            out.append(lp[np.arange(len(lp)), y[start : start + batch]])
        return np.concatenate(out)

    def memory_floats(self):
        return int(sum(v.size for v in self.params.values()))


class SparseCodeEncoder:
    """Fixed random sparse expansion of the one-hot context, followed by k-winner-take-all.

    Nothing here is learned; the projection is generated from a seed.
    """

    def __init__(self, vocab_size, context, units=2000, fan_in=8, k_active=64, seed=0):
        if not 0 < k_active <= units:
            raise ValueError("k_active must be in (0, units]")
        rng = np.random.default_rng(seed)
        self.vocab_size, self.context = vocab_size, context
        self.units, self.k_active = units, k_active
        n_in = context * vocab_size
        self.projection = np.zeros((units, n_in), np.float32)
        for u in range(units):
            self.projection[u, rng.choice(n_in, size=fan_in, replace=False)] = 1.0
        self._offsets = np.arange(context) * vocab_size

    def encode(self, ctx):
        B = len(ctx)
        onehot = np.zeros((B, self.context * self.vocab_size), np.float32)
        onehot[np.arange(B)[:, None], ctx + self._offsets] = 1.0
        response = onehot @ self.projection.T
        winners = np.argpartition(-response, self.k_active - 1, axis=1)[:, : self.k_active]
        codes = np.zeros((B, self.units), np.float32)
        np.put_along_axis(codes, winners, 1.0, axis=1)
        return codes

    def memory_floats(self):
        return int(self.projection.size)


class SparseCodeReadout:
    """Fixed sparse code + softmax readout trained by gradient descent (tests the code in a gradient setting)."""

    def __init__(self, vocab_size, context, units=2000, fan_in=8, k_active=64, seed=0):
        self.encoder = SparseCodeEncoder(vocab_size, context, units, fan_in, k_active, seed)
        self.params = {
            "W": np.zeros((units, vocab_size), np.float32),
            "b": np.zeros(vocab_size, np.float32),
        }

    def loss_and_grads(self, ctx, y):
        codes = self.encoder.encode(ctx)
        B = len(y)
        lp = log_softmax(codes @ self.params["W"] + self.params["b"])
        loss = -lp[np.arange(B), y].mean()
        dz = np.exp(lp)
        dz[np.arange(B), y] -= 1
        dz /= B
        return float(loss), {"W": codes.T @ dz, "b": dz.sum(0)}

    def fit(self, ctx, y, epochs, batch_size, lr, seed=0):
        opt = Adam(self.params, lr)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                b = order[start : start + batch_size]
                _, grads = self.loss_and_grads(ctx[b], y[b])
                opt.step(grads)

    def log_prob(self, ctx, y, batch=16384):
        out = []
        for start in range(0, len(y), batch):
            codes = self.encoder.encode(ctx[start : start + batch])
            lp = log_softmax(codes @ self.params["W"] + self.params["b"])
            out.append(lp[np.arange(len(lp)), y[start : start + batch]])
        return np.concatenate(out)

    def memory_floats(self):
        return int(self.params["W"].size + self.params["b"].size + self.encoder.memory_floats())


class PrototypeCharLM:
    """Gradient-free H1 adaptation: each next character gets an additive memory row of sparse codes.

    Training is one counting pass: M[y] += code(context). Prediction is cosine
    similarity to normalised rows, turned into probabilities with a temperature tau.
    """

    def __init__(self, vocab_size, context, units=2000, fan_in=8, k_active=64, seed=0):
        self.encoder = SparseCodeEncoder(vocab_size, context, units, fan_in, k_active, seed)
        self.vocab_size = vocab_size
        self.M = np.zeros((vocab_size, units), np.float64)

    def fit(self, ctx, y, batch=16384):
        for start in range(0, len(y), batch):
            codes = self.encoder.encode(ctx[start : start + batch])
            yb = y[start : start + batch]
            onehot = np.zeros((len(yb), self.vocab_size), np.float32)
            onehot[np.arange(len(yb)), yb] = 1.0
            self.M += (onehot.T @ codes).astype(np.float64)

    def scores(self, ctx):
        norms = np.linalg.norm(self.M, axis=1, keepdims=True)
        prototypes = (self.M / np.maximum(norms, 1e-12)).astype(np.float32)
        return self.encoder.encode(ctx) @ prototypes.T

    def log_prob(self, ctx, y, tau=0.1, batch=16384):
        out = []
        for start in range(0, len(y), batch):
            s = self.scores(ctx[start : start + batch]) / tau
            lp = log_softmax(s)
            out.append(lp[np.arange(len(lp)), y[start : start + batch]])
        return np.concatenate(out)

    def memory_floats(self):
        return int(self.M.size + self.encoder.memory_floats())
