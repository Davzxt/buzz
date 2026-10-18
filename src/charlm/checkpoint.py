"""Save/load a dense character model and sample text from it."""
import json

import numpy as np

from .data import CharVocab
from .models import DenseCharMLP
from .optim import log_softmax


def save_checkpoint(path, model, vocab, config):
    arrays = dict(model.params)
    arrays["vocab_chars"] = np.array(vocab.chars)
    arrays["config_json"] = np.array(json.dumps(config, ensure_ascii=False))
    np.savez(path, **arrays)


def load_checkpoint(path):
    data = np.load(path, allow_pickle=False)
    config = json.loads(str(data["config_json"]))
    vocab = CharVocab("".join(data["vocab_chars"].tolist()))
    model = DenseCharMLP(len(vocab), config["context"], config["emb"], config["hidden"])
    for name in model.params:
        model.params[name] = data[name].astype(np.float32)
    return model, vocab, config


def sample(model, vocab, config, prompt, max_new=300, temperature=0.8, stop="\n\n", rng=None):
    """Continue `prompt` one character at a time until `stop` appears or max_new is reached."""
    rng = rng or np.random.default_rng()
    context = config["context"]
    text = prompt
    generated = ""
    for _ in range(max_new):
        window = text[-context:]
        window = "".join(c if c in vocab.index else " " for c in window)
        window = " " * (context - len(window)) + window
        ids = vocab.encode(window)[None, :]
        _, _, logits = model._forward(ids)
        logits = log_softmax(logits / max(temperature, 1e-3))
        probs = np.exp(logits[0].astype(np.float64))
        probs /= probs.sum()
        char = vocab.chars[rng.choice(len(probs), p=probs)]
        text += char
        generated += char
        if stop and stop in generated:
            return generated.split(stop)[0].rstrip()
    return generated.strip()
