"""Baselines: nearest class mean, 1-NN, and a numpy softmax/MLP with optional replay."""
import numpy as np


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class NearestClassMean:
    """Stores one mean vector per class on raw features."""

    def __init__(self):
        self.labels = np.zeros(0, dtype=int)
        self.means = np.zeros((0, 0))

    def fit_task(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        new_labels = np.unique(y)
        new_means = np.stack([X[y == c].mean(axis=0) for c in new_labels])
        if self.means.size == 0:
            self.labels, self.means = new_labels, new_means
        else:
            self.labels = np.concatenate([self.labels, new_labels])
            self.means = np.vstack([self.means, new_means])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dist = ((X[:, None, :] - self.means[None, :, :]) ** 2).sum(axis=2)
        return self.labels[np.argmin(dist, axis=1)]

    def memory_floats(self):
        return int(self.means.size)


class NearestNeighbour1:
    """Stores every training example on raw features and returns the nearest label."""

    def __init__(self):
        self._X = []
        self._y = []

    def fit_task(self, X, y):
        self._X.append(np.asarray(X, dtype=float))
        self._y.append(np.asarray(y))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        stored_X, stored_y = np.vstack(self._X), np.concatenate(self._y)
        dist = (
            (X ** 2).sum(axis=1)[:, None]
            - 2.0 * X @ stored_X.T
            + (stored_X ** 2).sum(axis=1)[None, :]
        )
        return stored_y[np.argmin(dist, axis=1)]

    def memory_floats(self):
        return int(sum(X.size for X in self._X))


class IncrementalNet:
    """Softmax (hidden=0) or one-hidden-layer ReLU MLP with a growing output head.

    Trained with minibatch SGD on cross-entropy over all classes seen so far.
    replay_per_class=0 is naive sequential fine-tuning. A positive value keeps that
    many randomly chosen exemplars per class and mixes them into later tasks.
    """

    def __init__(
        self,
        input_dim,
        hidden=0,
        lr=0.1,
        epochs=20,
        batch_size=32,
        replay_per_class=0,
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.replay_per_class = replay_per_class
        feat_dim = hidden if hidden else input_dim
        if hidden:
            self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden))
            self.b1 = np.zeros(hidden)
        self.Wout = np.zeros((feat_dim, 0))
        self.bout = np.zeros(0)
        self.labels = []
        self._exemplars = {}

    def _features(self, X):
        if not self.hidden:
            return X, None
        Z = X @ self.W1 + self.b1
        return np.maximum(Z, 0.0), Z

    def _add_heads(self, new_labels):
        if not new_labels:
            return
        feat_dim = self.Wout.shape[0]
        self.Wout = np.hstack([self.Wout, np.zeros((feat_dim, len(new_labels)))])
        self.bout = np.concatenate([self.bout, np.zeros(len(new_labels))])
        self.labels.extend(new_labels)

    def _sgd_step(self, Xb, tb):
        F, Z = self._features(Xb)
        P = _softmax(F @ self.Wout + self.bout)
        P[np.arange(len(tb)), tb] -= 1.0
        G = P / len(tb)
        if self.hidden:
            dZ = (G @ self.Wout.T) * (Z > 0)
            gW1, gb1 = Xb.T @ dZ, dZ.sum(axis=0)
        gWout, gbout = F.T @ G, G.sum(axis=0)
        if self.hidden:
            self.W1 -= self.lr * gW1
            self.b1 -= self.lr * gb1
        self.Wout -= self.lr * gWout
        self.bout -= self.lr * gbout

    def fit_task(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        new_labels = [int(c) for c in np.unique(y)]
        self._add_heads([c for c in new_labels if c not in self.labels])

        Xs, ys = [X], [y]
        for c, Xc in self._exemplars.items():
            Xs.append(Xc)
            ys.append(np.full(len(Xc), c))
        Xt, yt = np.vstack(Xs), np.concatenate(ys)
        head = {c: i for i, c in enumerate(self.labels)}
        tt = np.array([head[int(c)] for c in yt])

        for _ in range(self.epochs):
            order = self.rng.permutation(len(Xt))
            for start in range(0, len(Xt), self.batch_size):
                batch = order[start : start + self.batch_size]
                self._sgd_step(Xt[batch], tt[batch])

        if self.replay_per_class:
            for c in new_labels:
                candidates = np.flatnonzero(y == c)
                take = min(self.replay_per_class, len(candidates))
                self._exemplars[c] = X[self.rng.choice(candidates, size=take, replace=False)]

    def predict(self, X):
        F, _ = self._features(np.asarray(X, dtype=float))
        logits = F @ self.Wout + self.bout
        return np.asarray(self.labels)[np.argmax(logits, axis=1)]

    def memory_floats(self):
        params = self.Wout.size + self.bout.size
        if self.hidden:
            params += self.W1.size + self.b1.size
        replay = sum(Xc.size for Xc in self._exemplars.values())
        return int(params + replay)
