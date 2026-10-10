#!/usr/bin/env bash
# Instalação/atualização na VPS (Ubuntu/Debian). Rode como root a partir de /opt/buzz.
set -euo pipefail
cd "$(dirname "$0")/.."
apt-get update -qq && apt-get install -y -qq python3-venv build-essential cmake curl
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
CMAKE_ARGS="-DGGML_NATIVE=ON" .venv/bin/pip install -q llama-cpp-python   # compila p/ a CPU da VPS (leva alguns min)
bash scripts/fetch_model.sh
[ -f models/decision.joblib ] || .venv/bin/python -m buzz.train
[ -f .env ] || { echo "BUZZ_API_KEY=$(head -c 24 /dev/urandom | base64 | tr -d '/+=')" > .env
  echo "BUZZ_MODEL_PATH=$(pwd)/models/qwen2.5-1.5b-instruct-q4_k_m.gguf" >> .env; chmod 600 .env; }
# swap de segurança (4 GB de RAM é apertado) — só cria se não houver
swapon --show | grep -q . || { fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile; }
sed "s#/opt/buzz#$(pwd)#g" scripts/buzz.service > /etc/systemd/system/buzz.service
systemctl daemon-reload && systemctl enable --now buzz && systemctl restart buzz
echo "ok: curl -s localhost:8000/health   (chave em $(pwd)/.env)"
