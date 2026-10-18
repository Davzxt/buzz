import numpy as np

from charlm import (DenseCharMLP, PrototypeCharLM, SparseCodeReadout, WittenBellNGram,
                    make_windows, log_softmax)


def test_make_windows_aligns_context_and_target():
    ids = np.arange(10)
    ctx, y = make_windows(ids, context=3)
    assert ctx.shape == (7, 3) and y.shape == (7,)
    assert ctx[0].tolist() == [0, 1, 2] and y[0] == 3
    assert ctx[-1].tolist() == [6, 7, 8] and y[-1] == 9


def test_log_softmax_rows_sum_to_one():
    z = np.random.default_rng(0).normal(size=(4, 7))
    assert np.allclose(np.exp(log_softmax(z)).sum(axis=1), 1.0)


def test_dense_mlp_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    model = DenseCharMLP(vocab_size=5, context=3, emb=4, hidden=6, seed=1)
    ctx = rng.integers(0, 5, size=(8, 3))
    y = rng.integers(0, 5, size=8)
    _, grads = model.loss_and_grads(ctx, y)
    eps = 1e-3
    for name in ["E", "W1", "b1", "W2", "b2"]:
        param = model.params[name]
        idx = tuple(rng.integers(0, s) for s in param.shape)
        old = param[idx]
        param[idx] = old + eps
        plus, _ = model.loss_and_grads(ctx, y)
        param[idx] = old - eps
        minus, _ = model.loss_and_grads(ctx, y)
        param[idx] = old
        numeric = (plus - minus) / (2 * eps)
        assert np.isclose(grads[name][idx], numeric, rtol=5e-2, atol=5e-4), name


def test_sparse_readout_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    model = SparseCodeReadout(vocab_size=6, context=3, units=64, fan_in=4, k_active=8, seed=2)
    model.params["W"] += rng.normal(0, 0.1, model.params["W"].shape).astype(np.float32)
    ctx = rng.integers(0, 6, size=(10, 3))
    y = rng.integers(0, 6, size=10)
    _, grads = model.loss_and_grads(ctx, y)
    eps = 1e-3
    W = model.params["W"]
    idx = (3, 2)
    old = W[idx]
    W[idx] = old + eps
    plus, _ = model.loss_and_grads(ctx, y)
    W[idx] = old - eps
    minus, _ = model.loss_and_grads(ctx, y)
    W[idx] = old
    assert np.isclose(grads["W"][idx], (plus - minus) / (2 * eps), rtol=5e-2, atol=5e-4)


def test_sparse_codes_have_exactly_k_bits():
    model = SparseCodeReadout(vocab_size=6, context=3, units=64, fan_in=4, k_active=8, seed=0)
    codes = model.encoder.encode(np.random.default_rng(0).integers(0, 6, size=(5, 3)))
    assert np.all(codes.sum(axis=1) == 8)


def test_ngram_distribution_sums_to_one_over_vocabulary():
    ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 3, 0, 1, 2] * 5)
    model = WittenBellNGram(order=3, vocab_size=4).fit(ids)
    ctx = np.array([[0, 1]] * 4)
    probs = [np.exp(model.log_prob(ctx, np.full(4, v))[0]) for v in range(4)]
    assert np.isclose(sum(probs), 1.0, atol=1e-9)


def test_prototype_lm_probabilities_are_normalised():
    rng = np.random.default_rng(0)
    ctx = rng.integers(0, 5, size=(200, 3))
    y = rng.integers(0, 5, size=200)
    model = PrototypeCharLM(vocab_size=5, context=3, units=128, fan_in=4, k_active=16, seed=0)
    model.fit(ctx, y)
    probs = [np.exp(model.log_prob(ctx[:1], np.array([v]), tau=0.1)[0]) for v in range(5)]
    assert np.isclose(sum(probs), 1.0, atol=1e-6)
