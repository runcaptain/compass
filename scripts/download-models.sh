#!/usr/bin/env bash
# Pre-deploy step to download BGE-small embedding weights into the
# persistent volume. Runs before the service starts.
# Idempotent — skips download if the model is already present.
#
# Uses plain curl so the runtime image needs no Python.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/app/data}"
BGE_DIR="$DATA_DIR/models/bge-small"
HF_REPO="${COMPASS_BGE_REPO:-BAAI/bge-small-en-v1.5}"
BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"

# Files candle_bge.rs expects: config.json, model.safetensors, tokenizer.json
FILES=("config.json" "model.safetensors" "tokenizer.json")

if [ -f "$BGE_DIR/model.safetensors" ] && [ -f "$BGE_DIR/config.json" ] && [ -f "$BGE_DIR/tokenizer.json" ]; then
    echo "[download-models] BGE-small already present at $BGE_DIR — skipping"
    exit 0
fi

echo "[download-models] Fetching $HF_REPO into $BGE_DIR"
mkdir -p "$BGE_DIR"

for f in "${FILES[@]}"; do
    if [ -f "$BGE_DIR/$f" ]; then
        echo "  $f already present, skipping"
        continue
    fi
    echo "  downloading $f"
    curl -fsSL --retry 3 --retry-delay 2 -o "$BGE_DIR/$f.tmp" "$BASE_URL/$f"
    mv "$BGE_DIR/$f.tmp" "$BGE_DIR/$f"
done

echo "[download-models] Done. Contents:"
ls -la "$BGE_DIR"
