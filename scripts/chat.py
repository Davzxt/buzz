"""Chat with a trained checkpoint in the terminal.

    python scripts/chat.py --checkpoint checkpoints/best.npz
    python scripts/chat.py --checkpoint checkpoints/best.npz --prompt "Dê três dicas de estudo."

The model continues the text after 'Assistente: ' one character at a time. It is a character
model trained on a small corpus, so expect short, repetitive or off-topic answers.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from charlm.checkpoint import load_checkpoint, sample  # noqa: E402


def reply(model, vocab, config, message, temperature, rng):
    prompt = f"Usuário: {message}\nAssistente: "
    return sample(model, vocab, config, prompt, temperature=temperature, rng=rng)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.npz")
    p.add_argument("--prompt", default=None, help="answer one message and exit")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, vocab, config = load_checkpoint(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    if args.prompt is not None:
        print(reply(model, vocab, config, args.prompt, args.temperature, rng))
        return
    print("Chat de caractere (digite 'sair' para encerrar).")
    while True:
        try:
            message = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"sair", "exit", "quit"}:
            break
        if message:
            print("Modelo:", reply(model, vocab, config, message, args.temperature, rng))


if __name__ == "__main__":
    main()
