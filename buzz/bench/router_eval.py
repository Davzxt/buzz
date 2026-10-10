"""Avalia o caminho de decisão (precisão, calibração, latência) em templates NÃO vistos.
    python -m bench.router_eval
"""
import json
import statistics
import time

from buzz.data import splits
from buzz.decision import Decision
from buzz.train import evaluate

d = Decision.load("models/decision.joblib")
_, _, test = splits(150)
res = evaluate(d, test)
lat = []
for r in test:
    t = time.perf_counter()
    d.predict(r["text"])
    lat.append((time.perf_counter() - t) * 1000)
lat.sort()
res["decision_ms_p50"] = round(statistics.median(lat), 3)
res["decision_ms_p95"] = round(lat[int(.95 * len(lat))], 3)
direct = sum(r["label"] in ("math", "remember") for r in test) / len(test)
res["share_answered_without_llm_if_perfect"] = round(direct, 3)
print(json.dumps(res, indent=2))
