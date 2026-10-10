#!/usr/bin/env bash
# Baixa um GGUF quantizado (Apache-2.0). Padrão: Qwen2.5-1.5B-Instruct Q4_K_M (~1.1 GB).
# Mais leve/rápido: BUZZ_SIZE=0.5b  (~0.4 GB, qualidade menor)
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE="${BUZZ_SIZE:-1.5b}"
F="qwen2.5-${SIZE}-instruct-q4_k_m.gguf"
mkdir -p models
[ -f "models/$F" ] || curl -L --fail -sS -o "models/$F" "https://huggingface.co/Qwen/Qwen2.5-$(echo "$SIZE" | tr a-z A-Z)-Instruct-GGUF/resolve/main/$F"
echo "BUZZ_MODEL_PATH=$(pwd)/models/$F"
