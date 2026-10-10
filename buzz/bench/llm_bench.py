"""Benchmark do núcleo linguístico NA VPS: RAM de pico, TTFT, tokens/s, latência p50/p95.
    python -m bench.llm_bench [--runs 3]
Resultados reais só valem se rodados no KVM 1. Nada aqui é estimado.
"""
import argparse
import json
import statistics
import time

import psutil

from buzz.llm import LLM

PROMPTS = ["Explique em duas frases o que é inflação.", "Resuma: O gato dorme o dia todo e caça à noite pelos telhados do bairro.",
           "Escreva uma saudação curta e simpática.", "Quais as capitais do Sul do Brasil?", "Qual a diferença entre RAM e SSD?"]
ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=3)
a = ap.parse_args()
llm, proc = LLM(), psutil.Process()
t0 = time.time()
llm._model()
load_s = time.time() - t0
lat, tps, peak = [], [], proc.memory_info().rss
for _ in range(a.runs):
    for p in PROMPTS:
        t = time.time()
        _, u = llm.chat([{"role": "user", "content": p}], max_tokens=96, temperature=0.2)
        dt = time.time() - t
        lat.append(dt)
        tps.append(u.get("completion_tokens", 0) / dt)
        peak = max(peak, proc.memory_info().rss)
lat.sort()
print(json.dumps({"model": llm.path, "load_s": round(load_s, 1), "rss_mb_peak": round(peak / 2**20),
                  "latency_s_p50": round(statistics.median(lat), 2), "latency_s_p95": round(lat[int(.95 * len(lat)) - 1], 2),
                  "tokens_per_s_median": round(statistics.median(tps), 1), "n": len(lat)}, indent=2))
