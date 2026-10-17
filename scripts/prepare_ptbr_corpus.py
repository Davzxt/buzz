"""Build one mixed corpus: Portuguese conversation/instructions, Portuguese Wikipedia, and Python code.

Every source is read from files on the Hugging Face Hub (parquet or JSON Lines), so no
dataset loading scripts are needed. Each source gets a character budget from `share`.
Licenses are recorded in the manifest. Run from the repository root:

    python scripts/prepare_ptbr_corpus.py
"""
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

OUTPUT = Path("data/ptbr_mixed_corpus.txt")
SEPARATOR = "\n\n"
CHARS_TOTAL = 6_000_000

SOURCES = [
    {
        "name": "conversation_ptbr",
        "dataset": "dominguesm/alpaca-data-pt-br",
        "license": "cc-by-nc-4.0 (non-commercial)",
        "share": 0.35,
        "files": ["data/train-00000-of-00001-63604ec441480f0e.parquet"],
        "format": lambda r: f"Usuário: {r['instruction']}\n{r['input']}\nAssistente: {r['output']}".strip(),
    },
    {
        "name": "education_ptbr",
        "dataset": "wikimedia/wikipedia",
        "license": "cc-by-sa-3.0 / gfdl",
        "share": 0.35,
        "files": "wikipedia_pt",
        "format": lambda r: r["text"],
    },
    {
        "name": "code_python",
        "dataset": "bigcode/the-stack-smol-xs",
        "license": "per-file (The Stack); verify before redistribution",
        "share": 0.30,
        "hub_file": "data/python/data.json",
        "format": lambda r: r["content"],
    },
]


def wikipedia_pt_files():
    siblings = HfApi().dataset_info("wikimedia/wikipedia").siblings
    return sorted(s.rfilename for s in siblings if s.rfilename.startswith("20231101.pt/"))


def iter_rows(src):
    if src.get("hub_file"):
        path = hf_hub_download(src["dataset"], src["hub_file"], repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    files = wikipedia_pt_files() if src["files"] == "wikipedia_pt" else src["files"]
    for name in files:
        path = hf_hub_download(src["dataset"], name, repo_type="dataset")
        for row in pq.read_table(path).to_pylist():
            yield row


def collect(src, budget):
    pieces, used = [], 0
    for row in iter_rows(src):
        text = src["format"](row).strip()
        if not text:
            continue
        pieces.append(text)
        used += len(text) + len(SEPARATOR)
        if used >= budget:
            break
    return pieces, used


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"output": str(OUTPUT), "sources": []}
    with OUTPUT.open("w", encoding="utf-8") as out:
        for src in SOURCES:
            pieces, used = collect(src, int(CHARS_TOTAL * src["share"]))
            out.write(SEPARATOR.join(pieces) + SEPARATOR)
            manifest["sources"].append({"name": src["name"], "dataset": src["dataset"],
                                        "license": src["license"], "records": len(pieces), "chars": used})
            print(f"{src['name']}: {len(pieces)} records, {used} chars", flush=True)
    Path(str(OUTPUT) + ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
