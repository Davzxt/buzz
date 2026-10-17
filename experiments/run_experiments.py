"""Reproducible experiment suite for hypothesis H1.

Writes results/results.json, results/summary.js (read by frontend/index.html)
and results/report.md. Run from the repository root:

    python experiments/run_experiments.py --config configs/experiment.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import IncrementalNet, NearestClassMean, NearestNeighbour1  # noqa: E402
from data import (  # noqa: E402
    class_incremental_tasks,
    load_digits_split,
    sample_few_shot,
    split_validation,
)
from metrics import accuracy, class_incremental_metrics  # noqa: E402
from prototype_memory import SparsePrototypeClassifier  # noqa: E402

INPUT_DIM = 64
N_CLASSES = 10


def _mlp_factory(cfg, seed, replay=0):
    return IncrementalNet(
        INPUT_DIM,
        hidden=cfg["hidden"],
        replay_per_class=replay,
        seed=seed,
        **cfg["sgd"],
    )


def class_incremental_methods(cfg, seed):
    return {
        "prototype_sparse": lambda: SparsePrototypeClassifier(INPUT_DIM, **cfg["prototype"], seed=seed),
        "ncm_raw": NearestClassMean,
        "naive_softmax": lambda: IncrementalNet(INPUT_DIM, hidden=0, seed=seed, **cfg["sgd"]),
        "naive_mlp": lambda: _mlp_factory(cfg, seed, replay=0),
        "replay_mlp": lambda: _mlp_factory(cfg, seed, replay=cfg["replay_per_class"]),
    }


def run_class_incremental_once(cfg, seed):
    Xtr, ytr, Xte, yte = load_digits_split(seed, cfg["test_size"])
    tasks = class_incremental_tasks(cfg["classes_per_task"], N_CLASSES)
    results = {}

    for name, make in class_incremental_methods(cfg, seed).items():
        model = make()
        T = len(tasks)
        acc_matrix = np.full((T, T), np.nan)
        overall, fit_time, old_drift = [], 0.0, 0.0
        seen = []
        for t, task_classes in enumerate(tasks):
            seen += task_classes
            in_task = np.isin(ytr, task_classes)
            prev = model.memory.snapshot() if hasattr(model, "memory") and model.memory.labels else None
            start = time.perf_counter()
            model.fit_task(Xtr[in_task], ytr[in_task])
            fit_time += time.perf_counter() - start
            if prev is not None:
                now = model.memory.snapshot()
                old_drift = max(old_drift, float(np.max(np.abs(now[: len(prev)] - prev))))
            seen_mask = np.isin(yte, seen)
            pred_seen = model.predict(Xte[seen_mask])
            overall.append(accuracy(yte[seen_mask], pred_seen))
            for s in range(t + 1):
                mask = np.isin(yte, tasks[s])
                acc_matrix[t, s] = accuracy(yte[mask], model.predict(Xte[mask]))

        start = time.perf_counter()
        model.predict(Xte)
        predict_ms = (time.perf_counter() - start) * 1000 / len(Xte)

        metrics = class_incremental_metrics(acc_matrix, overall)
        metrics.update(
            fit_time_s=fit_time,
            predict_ms_per_sample=predict_ms,
            memory_floats=model.memory_floats(),
            old_prototype_max_drift=old_drift,
        )
        results[name] = metrics

    joint = MLPClassifier(random_state=seed, **cfg["joint_mlp"])
    start = time.perf_counter()
    joint.fit(Xtr, ytr)
    results["joint_mlp_offline"] = {
        "final_acc": accuracy(yte, joint.predict(Xte)),
        "fit_time_s": time.perf_counter() - start,
        "note": "trained once on all classes; not a continual method, upper-bound reference",
    }
    return results


def run_few_shot_once(cfg, seed, k):
    Xtr, ytr, Xte, yte = load_digits_split(seed, cfg["test_size"])
    rng = np.random.default_rng(seed * 1000 + k)
    idx = sample_few_shot(ytr, k, range(N_CLASSES), rng)
    Xk, yk = Xtr[idx], ytr[idx]

    sparse = SparsePrototypeClassifier(INPUT_DIM, **cfg["prototype"], seed=seed)
    ncm, nn1 = NearestClassMean(), NearestNeighbour1()
    logreg = LogisticRegression(max_iter=2000)
    mlp = MLPClassifier(random_state=seed, **cfg["joint_mlp"])

    sparse.fit_task(Xk, yk)
    ncm.fit_task(Xk, yk)
    nn1.fit_task(Xk, yk)
    logreg.fit(Xk, yk)
    mlp.fit(Xk, yk)

    return {
        "prototype_sparse": accuracy(yte, sparse.predict(Xte)),
        "ncm_raw": accuracy(yte, ncm.predict(Xte)),
        "nn1_raw": accuracy(yte, nn1.predict(Xte)),
        "logreg": accuracy(yte, logreg.predict(Xte)),
        "mlp_fewshot": accuracy(yte, mlp.predict(Xte)),
    }


def run_ablation_once(cfg, seed):
    Xtr, ytr, _, _ = load_digits_split(seed, cfg["test_size"])
    Xa, ya, Xv, yv = split_validation(Xtr, ytr, seed)
    default = dict(cfg["prototype"])
    rows = []
    for param, values in cfg["ablation"].items():
        for value in values:
            params = dict(default, **{param: value})
            if params["k_active"] > params["expansion_dim"]:
                continue
            clf = SparsePrototypeClassifier(INPUT_DIM, **params, seed=seed)
            clf.fit_task(Xa, ya)
            rows.append({
                "param": param,
                "value": value,
                "val_acc": accuracy(yv, clf.predict(Xv)),
                "memory_floats": clf.memory_floats(),
            })
    return rows


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def aggregate_class_incremental(per_seed):
    methods = per_seed[0].keys()
    summary = {}
    for m in methods:
        summary[m] = {}
        for key in per_seed[0][m]:
            if key == "note":
                summary[m][key] = per_seed[0][m][key]
                continue
            values = [run[m][key] for run in per_seed]
            mean, std = mean_std(values)
            summary[m][key] = {"mean": mean, "std": std, "values": values}
    return summary


def aggregate_few_shot(per_k):
    out = {}
    for k, runs in per_k.items():
        out[k] = {}
        for m in runs[0]:
            values = [r[m] for r in runs]
            mean, std = mean_std(values)
            out[k][m] = {"mean": mean, "std": std, "values": values}
    return out


def aggregate_ablation(per_seed):
    table = {}
    for run in per_seed:
        for row in run:
            key = f"{row['param']}={row['value']}"
            table.setdefault(key, {"param": row["param"], "value": row["value"], "val": [], "mem": row["memory_floats"]})
            table[key]["val"].append(row["val_acc"])
    rows = []
    for entry in table.values():
        mean, std = mean_std(entry["val"])
        rows.append({"param": entry["param"], "value": entry["value"], "val_acc_mean": mean,
                     "val_acc_std": std, "memory_floats": entry["mem"]})
    return rows


def _judge(diff, p, margin=0.0, alpha=0.05):
    """Paired comparison over seeds. Three states: supported, refuted, inconclusive."""
    if p < alpha and diff > margin:
        return "SUSTENTADO"
    if p < alpha and diff < 0:
        return "NÃO SUSTENTADO"
    return "INCONCLUSIVO"


def _paired(a, b):
    """Mean of (a - b) over seeds and the two-sided paired t-test p-value."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = float(np.mean(a - b))
    p = 1.0 if np.allclose(a, b) else float(ttest_rel(a, b).pvalue)
    return diff, p


def verdicts(ci, fs):
    def vals(name, key="final_acc"):
        return ci[name][key]["values"]

    proto_drift = max(vals("prototype_sparse", "old_prototype_max_drift"))
    naive_best = np.maximum(vals("naive_softmax"), vals("naive_mlp"))
    out = {
        "H1a_no_drift_of_stored_prototypes": {
            "status": "SUSTENTADO" if proto_drift == 0.0 else "NÃO SUSTENTADO",
            "evidence": f"max |Δ| of old prototype rows over all seeds and tasks = {proto_drift:.3g}",
        },
    }

    d, p = _paired(vals("prototype_sparse"), naive_best)
    out["H1b_beats_best_naive_incremental_by_5pts"] = {
        "status": _judge(d, p, margin=0.05),
        "evidence": f"Δ = {d:+.3f}, paired t-test p = {p:.2g} (n={len(naive_best)} seeds)",
    }

    d, p = _paired(vals("prototype_sparse"), vals("replay_mlp"))
    out["H1c_matches_or_beats_replay_mlp"] = {
        "status": _judge(d, p),
        "evidence": f"Δ = {d:+.3f}, paired t-test p = {p:.2g}",
    }

    few_parts, few_status = [], []
    for k in ("1", "5"):
        best_baseline = np.maximum(fs[k]["ncm_raw"]["values"], fs[k]["nn1_raw"]["values"])
        d, p = _paired(fs[k]["prototype_sparse"]["values"], best_baseline)
        few_status.append(_judge(d, p))
        few_parts.append(f"k={k}: Δ = {d:+.3f}, p = {p:.2g}")
    out["H1d_fewshot_beats_ncm_and_1nn_at_k1_k5"] = {
        "status": ("SUSTENTADO" if all(s == "SUSTENTADO" for s in few_status)
                   else "NÃO SUSTENTADO" if "NÃO SUSTENTADO" in few_status else "INCONCLUSIVO"),
        "evidence": "; ".join(few_parts) + " (vs best of NCM / 1-NN, paired over seeds)",
    }

    d, p = _paired(vals("replay_mlp", "fit_time_s"), vals("prototype_sparse", "fit_time_s"))
    out["H1e_fit_faster_than_replay_mlp"] = {
        "status": _judge(d, p),
        "evidence": f"Δ (replay − prototype) fit time = {d:+.3f} s, paired t-test p = {p:.2g}",
    }
    return out


def write_report(out_dir, results):
    ci, fs, ab, vd = results["class_incremental"], results["few_shot"], results["ablation"], results["verdicts"]
    lines = ["# Relatório de experimentos (gerado automaticamente)", "",
             f"Seeds: {results['config']['seeds']}. Média ± desvio-padrão entre seeds.", "",
             "## Aprendizado incremental de classes (digits, 5 tarefas de 2 classes)", "",
             "| Método | Acurácia final | Acurácia incremental média | Esquecimento | Tempo de treino (s) | Memória (floats) |",
             "|---|---|---|---|---|---|"]
    def cell(stats, fmt="{:.3f}"):
        return "—" if not isinstance(stats, dict) else fmt.format(stats["mean"])

    for name, s in ci.items():
        if "final_acc" not in s:
            continue
        spread = f" ± {s['final_acc']['std']:.3f}"
        lines.append(
            f"| {name} | {cell(s['final_acc'])}{spread} | {cell(s.get('avg_incremental_acc'))} | "
            f"{cell(s.get('forgetting'))} | {cell(s.get('fit_time_s'))} | {cell(s.get('memory_floats'), '{:.0f}')} |"
        )
    lines += ["", "## Poucos exemplos (acurácia no teste completo, k exemplos por classe)", "",
              "| k | " + " | ".join(next(iter(fs.values())).keys()) + " |",
              "|---" * (1 + len(next(iter(fs.values())))) + "|"]
    for k, row in fs.items():
        lines.append(f"| {k} | " + " | ".join(f"{v['mean']:.3f} ± {v['std']:.3f}" for v in row.values()) + " |")
    lines += ["", "## Ablação (acurácia de validação, sem uso do conjunto de teste)", "",
              "| Parâmetro | Valor | Acurácia val. | Memória (floats) |", "|---|---|---|---|"]
    for r in ab:
        lines.append(f"| {r['param']} | {r['value']} | {r['val_acc_mean']:.3f} ± {r['val_acc_std']:.3f} | {r['memory_floats']} |")
    lines += ["", "## Veredito dos critérios pré-registrados (docs/hypotheses.md)", ""]
    for key, v in vd.items():
        lines.append(f"- **{key}**: {v['status']}. {v['evidence']}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ci_runs = [run_class_incremental_once(cfg, s) for s in cfg["seeds"]]
    ci = aggregate_class_incremental(ci_runs)
    print(f"class-incremental done in {time.perf_counter() - t0:.1f}s", flush=True)

    fs_runs = {str(k): [run_few_shot_once(cfg, s, k) for s in cfg["seeds"]] for k in cfg["few_shot_k"]}
    fs = aggregate_few_shot(fs_runs)
    print("few-shot done", flush=True)

    ab = aggregate_ablation([run_ablation_once(cfg, s) for s in cfg["seeds"]])
    vd = verdicts(ci, fs)

    results = {"config": cfg, "class_incremental": ci, "few_shot": fs, "ablation": ab, "verdicts": vd,
               "elapsed_s": time.perf_counter() - t0}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.js").write_text("window.RESULTS = " + json.dumps(results, ensure_ascii=False) + ";\n",
                                        encoding="utf-8")
    write_report(out_dir, results)
    print(f"wrote {out_dir / 'results.json'} ({results['elapsed_s']:.1f}s total)")


if __name__ == "__main__":
    main()
