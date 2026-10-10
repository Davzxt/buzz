"""    python -m lm.sample --run runs/buzz-lm --prompt "A fotossíntese é" """
import argparse
import time

import torch
from tokenizers import Tokenizer

from .model import load

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/buzz-lm")
ap.add_argument("--prompt", default="O Brasil é")
ap.add_argument("--max-new", type=int, default=100)
ap.add_argument("--temp", type=float, default=0.8)
a = ap.parse_args()
tok = Tokenizer.from_file(f"{a.run}/tokenizer.json")
m = load(f"{a.run}/best.pt")
ids = torch.tensor([tok.encode(a.prompt).ids])
t = time.time()
out = m.generate(ids, a.max_new, a.temp, stop_id=tok.token_to_id("<eos>"))
dt = time.time() - t
print(a.prompt + tok.decode(out).replace(" ¶ ", "\n"))
print(f"\n[{len(out) / dt:.1f} tokens/s na CPU, {m.n_params() / 1e6:.1f}M params]")
