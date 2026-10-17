import numpy as np

from baselines import IncrementalNet, NearestClassMean, NearestNeighbour1
from data import class_incremental_tasks, load_digits_split, sample_few_shot
from metrics import class_incremental_metrics


def _two_blobs(seed=0, n=60):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0.2, 0.05, (n, 4)), rng.normal(0.8, 0.05, (n, 4))])
    y = np.repeat([0, 1], n)
    return X, y


def test_nearest_class_mean_and_1nn_solve_separable_blobs():
    X, y = _two_blobs()
    for model in (NearestClassMean(), NearestNeighbour1()):
        model.fit_task(X, y)
        assert np.mean(model.predict(X) == y) == 1.0


def test_nearest_class_mean_adds_classes_incrementally():
    X, y = _two_blobs()
    model = NearestClassMean()
    model.fit_task(X[y == 0], y[y == 0])
    model.fit_task(X[y == 1], y[y == 1])
    assert np.mean(model.predict(X) == y) == 1.0


def test_incremental_net_grows_head_and_learns_blobs():
    X, y = _two_blobs()
    net = IncrementalNet(4, hidden=0, lr=0.5, epochs=30, seed=0)
    net.fit_task(X, y)
    assert net.Wout.shape == (4, 2)
    assert np.mean(net.predict(X) == y) > 0.95


def test_replay_keeps_bounded_exemplars_per_class():
    X, y = _two_blobs(n=40)
    net = IncrementalNet(4, hidden=8, epochs=2, replay_per_class=5, seed=0)
    net.fit_task(X[y == 0], y[y == 0])
    net.fit_task(X[y == 1], y[y == 1])
    assert {c: len(v) for c, v in net._exemplars.items()} == {0: 5, 1: 5}


def test_naive_finetuning_forgets_the_first_class():
    X, y = _two_blobs()
    net = IncrementalNet(4, hidden=0, lr=0.5, epochs=30, seed=0)
    net.fit_task(X[y == 0], y[y == 0])
    assert np.mean(net.predict(X[y == 0]) == 0) > 0.9
    net.fit_task(X[y == 1], y[y == 1])
    assert np.mean(net.predict(X[y == 0]) == 0) < 0.1


def test_class_incremental_metrics_on_known_matrix():
    A = np.array([[0.9, np.nan], [0.6, 0.8]])
    overall = [0.9, 0.7]
    m = class_incremental_metrics(A, overall)
    assert np.isclose(m["final_acc"], 0.7)
    assert np.isclose(m["avg_incremental_acc"], 0.8)
    assert np.isclose(m["forgetting"], 0.3)


def test_task_partition_covers_all_classes_once():
    tasks = class_incremental_tasks(classes_per_task=2, n_classes=10)
    flat = [c for t in tasks for c in t]
    assert sorted(flat) == list(range(10)) and len(tasks) == 5


def test_few_shot_sampler_returns_k_per_class():
    _, ytr, _, _ = load_digits_split(seed=0)
    idx = sample_few_shot(ytr, k=3, classes=range(10), rng=np.random.default_rng(0))
    counts = np.bincount(ytr[idx], minlength=10)
    assert np.all(counts == 3)


def test_digits_split_has_expected_shape():
    Xtr, ytr, Xte, yte = load_digits_split(seed=0)
    assert Xtr.shape[1] == 64 and Xte.shape[1] == 64
    assert len(Xtr) + len(Xte) == 1797
    assert Xtr.max() <= 1.0 and Xtr.min() >= 0.0
