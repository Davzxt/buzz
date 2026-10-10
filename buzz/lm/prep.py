"""Corpus -> tokenizador BPE (treinado por nós, do zero) -> tokens binários.

    python -m lm.prep --docs 3000 --vocab 4000 --out data/lm_smoke      # teste rápido
    python -m lm.prep --docs 600000 --vocab 16000 --out data/lm         # treino real (Colab)

Fonte padrão: Wikipédia em português (wikimedia/wikipedia, 20231101.pt; licença CC BY-SA).
Para somar outros textos seus: --extra arquivo.txt (um documento por linha).
"""
import argparse
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["<pad>", "<eos>", "<usuario>", "<buzz>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=3000)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--out", default="data/lm")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "corpus.txt"

    if not raw.exists():
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
        with raw.open("w", encoding="utf-8") as f:
            for i, r in enumerate(ds):
                if i >= a.docs:
                    break
                t = re.sub(r"\s*\n\s*", " ¶ ", r["text"].strip())[:20000]
                if len(t) > 200:
                    f.write(t + "\n")
            for p in a.extra:
                for line in open(p, encoding="utf-8"):
                    f.write(line.strip()[:20000] + "\n")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator((l for l in open(raw, encoding="utf-8")),
                            trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=SPECIAL,
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(out / "tokenizer.json"))
    eos = tok.token_to_id("<eos>")

    parts, buf = [], []

    def flush():
        for e in tok.encode_batch(buf):
            parts.append(np.array(e.ids + [eos], dtype=np.uint16))
        buf.clear()
    for line in open(raw, encoding="utf-8"):  # streaming: RAM baixa mesmo com corpus grande
        buf.append(line.rstrip("\n"))
        if len(buf) == 512:
            flush()
    flush()
    arr = np.concatenate(parts)
    cut = int(len(arr) * 0.995)
    arr[:cut].tofile(out / "train.bin")
    arr[cut:].tofile(out / "val.bin")
    print(f"tokens: train={cut:,} val={len(arr) - cut:,} vocab={tok.get_vocab_size()}")


if __name__ == "__main__":
    main()
