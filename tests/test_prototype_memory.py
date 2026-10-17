import numpy as np
import pytest

from prototype_memory import PrototypeMemory, SparseExpansionEncoder, SparsePrototypeClassifier


def test_encoder_produces_exactly_k_active_bits():
    enc = SparseExpansionEncoder(input_dim=20, expansion_dim=200, fan_in=5, k_active=13, seed=1)
    codes = enc.encode(np.random.default_rng(0).random((7, 20)))
    assert codes.shape == (7, 200)
    assert set(np.unique(codes)) <= {0.0, 1.0}
    assert np.all(codes.sum(axis=1) == 13)


def test_encoder_is_deterministic_for_a_seed():
    X = np.random.default_rng(0).random((5, 16))
    a = SparseExpansionEncoder(16, expansion_dim=256, k_active=8, seed=3).encode(X)
    b = SparseExpansionEncoder(16, expansion_dim=256, k_active=8, seed=3).encode(X)
    c = SparseExpansionEncoder(16, expansion_dim=256, k_active=8, seed=4).encode(X)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_encoder_projection_has_fan_in_per_unit():
    enc = SparseExpansionEncoder(30, expansion_dim=50, fan_in=6, k_active=5, seed=0)
    assert np.all(enc.projection.sum(axis=1) == 6)


def test_encoder_rejects_bad_shapes_and_parameters():
    enc = SparseExpansionEncoder(10, expansion_dim=32, k_active=4, seed=0)
    with pytest.raises(ValueError):
        enc.encode(np.zeros((3, 9)))
    with pytest.raises(ValueError):
        SparseExpansionEncoder(10, expansion_dim=32, k_active=33)


def test_learning_a_new_class_does_not_change_old_prototypes():
    rng = np.random.default_rng(0)
    mem = PrototypeMemory(code_dim=40)
    mem.learn(rng.integers(0, 2, (5, 40)).astype(float), label=0)
    mem.learn(rng.integers(0, 2, (5, 40)).astype(float), label=1)
    before = mem.snapshot()
    mem.learn(rng.integers(0, 2, (9, 40)).astype(float), label=2)
    after = mem.snapshot()
    assert np.array_equal(after[:2], before)


def test_learning_order_does_not_matter():
    rng = np.random.default_rng(1)
    batches = [(rng.integers(0, 2, (4, 30)).astype(float), lab) for lab in [0, 1, 0, 2, 1]]
    forward, backward = PrototypeMemory(30), PrototypeMemory(30)
    for codes, lab in batches:
        forward.learn(codes, lab)
    for codes, lab in reversed(batches):
        backward.learn(codes, lab)
    assert np.allclose(forward.snapshot()[np.argsort(forward.labels)],
                       backward.snapshot()[np.argsort(backward.labels)])


def test_predict_returns_class_whose_prototype_matches_query():
    mem = PrototypeMemory(code_dim=6)
    mem.learn(np.array([[1, 1, 0, 0, 0, 0]], dtype=float), label=7)
    mem.learn(np.array([[0, 0, 0, 0, 1, 1]], dtype=float), label=9)
    preds = mem.predict(np.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=float))
    assert preds.tolist() == [7, 9]


def test_empty_memory_refuses_to_predict():
    with pytest.raises(RuntimeError):
        PrototypeMemory(code_dim=4).predict(np.zeros((1, 4)))


def test_classifier_learns_separable_classes_in_one_pass():
    rng = np.random.default_rng(0)
    centers = rng.random((3, 20))
    X = np.vstack([c + 0.02 * rng.standard_normal((30, 20)) for c in centers])
    y = np.repeat([0, 1, 2], 30)
    clf = SparsePrototypeClassifier(20, expansion_dim=500, k_active=40, seed=0)
    clf.fit_task(X, y)
    assert np.mean(clf.predict(X) == y) > 0.95
