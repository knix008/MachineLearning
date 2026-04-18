#!/usr/bin/env bash
# Optional first arg: git tag or branch to checkout (e.g. v0.24.0)
set -euo pipefail
TAG="${1:-}"
RAG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$RAG_DIR/ragflow"

checkout_ref() {
  local dir="$1" ref="$2"
  [[ -z "$ref" ]] && return 0
  echo "[bootstrap] Checking out $ref in $dir"
  git -C "$dir" fetch --tags origin 2>/dev/null || true
  git -C "$dir" checkout "$ref"
}

if [[ -d "$TARGET" ]]; then
  echo "[bootstrap] Already exists: $TARGET"
  checkout_ref "$TARGET" "$TAG"
  exit 0
fi

echo "[bootstrap] Cloning RAGFlow -> $TARGET"
git clone https://github.com/infiniflow/ragflow.git "$TARGET"
checkout_ref "$TARGET" "$TAG"
echo "[bootstrap] Done. Optional: cp config/env.override.example config/env.override && ./scripts/apply-overrides.sh"
echo "[bootstrap] Start: ./scripts/up.sh or ./scripts/up-gpu.sh"
