"""GSM8K (OpenAI, 1319 problemas de matemática) — subconjunto fixo, zero-shot, mesma máquina/threads.
Compara: (A) LLM puro  vs  (B) BUZZ completo (decisão + roteador + ferramentas).
    python -m bench.gsm8k --n 50 --seed 0
Observação: GSM8K é em inglês; mede raciocínio do núcleo, não o português do Buzz.
"""
import argparse
import json
import random
import re
import statistics
import time

from buzz.decision import Decision
from buzz.llm import LLM
from buzz.memory import Memory
from buzz.router import Router

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=320)
a = ap.parse_args()

rows = [json.loads(l) for l in open("data/gsm8k_test.jsonl")]
random.Random(a.seed).shuffle(rows)
rows = rows[:a.n]
SUFFIX = "\nSolve step by step, then end with a final line: 'Answer: <number>'."


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return float(m[-1].replace(",", "")) if m else None


def final(text):
    m = re.search(r"answer\s*[:=]?\s*\**\s*\$?(-?\d[\d,]*\.?\d*)", text, re.I)
    return float(m.group(1).replace(",", "")) if m else num(text)


llm = LLM()
router = Router(Decision.load("models/decision.joblib"), Memory(), llm)
out = {"A_llm_only": [], "B_buzz": []}
meta = {"A_llm_only": [], "B_buzz": []}
for i, r in enumerate(rows):
    gold = num(r["answer"].split("####")[-1])
    q = r["question"] + SUFFIX
    t = time.time()
    ta, _ = llm.chat([{"role": "user", "content": q}], a.max_tokens, 0.0)
    out["A_llm_only"].append((final(ta) == gold, time.time() - t))
    t = time.time()
    tb, m = router.handle([{"role": "user", "content": q}], a.max_tokens, 0.0)
    out["B_buzz"].append((final(tb) == gold, time.time() - t))
    meta["B_buzz"].append(m["route"])
    print(i + 1, out["A_llm_only"][-1][0], out["B_buzz"][-1][0], m["route"], flush=True)

res = {"benchmark": "GSM8K test (subset aleatório fixo)", "n": a.n, "seed": a.seed, "model": llm.path, "threads": llm.n_threads}
for k, v in out.items():
    ok = sum(x for x, _ in v)
    res[k] = {"acc": round(ok / len(v), 3), "correct": ok, "latency_s_p50": round(statistics.median(t for _, t in v), 1)}
res["B_routes"] = {r: meta["B_buzz"].count(r) for r in set(meta["B_buzz"])}
print(json.dumps(res, indent=2))
open("docs/gsm8k_result.json", "w").write(json.dumps(res, indent=2))
