"""Build a mixed conversation / education / code corpus from Hugging Face streams.

Reads configs/datasets.json, streams each source, keeps records matching its filter,
and writes sources in proportion to `share` until each source's character budget is met.
Each source's license is recorded in the output manifest.

    python scripts/prepare_mixed_corpus.py --config configs/datasets.json
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def iter_rows(src):
    if src.get("hub_file"):
        path = hf_hub_download(src["dataset"], src["hub_file"], repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from load_dataset(src["dataset"], src["config"], split=src["split"], streaming=True)


def stream_source(src, budget_chars, separator):
    pieces, used = [], 0
    for row in iter_rows(src):
        if any(row.get(k) != v for k, v in src["filter"].items()):
            continue
        text_field = src["fields"]["text"]
        text = (row.get(text_field) or "").strip()
        if not text:
            continue
        fmt = src["format"]
        piece = fmt.format(text=text, role=row.get("role", "")).strip()
        pieces.append(piece)
        used += len(piece) + len(separator)
        if used >= budget_chars:
            break
    return separator.join(pieces), used, len(pieces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": []}
    with out.open("w", encoding="utf-8") as f:
        for i, src in enumerate(cfg["sources"]):
            budget = int(cfg["chars_total"] * src["share"])
            text, used, n = stream_source(src, budget, cfg["separator"])
            f.write(text + cfg["separator"])
            manifest["sources"].append({"name": src["name"], "dataset": src["dataset"], "config": src.get("config"),
                                        "license": src["license"], "records": n, "chars": used})
            print(f"{src['name']}: {n} records, {used} chars", flush=True)
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
