#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
while [[ ! -d .git ]] && [[ "$(pwd)" != "/" ]]; do cd ..; done
if [[ ! -d .git ]]; then
  echo "No .git directory found above scripts folder." >&2
  exit 1
fi
ROOT="$(pwd)"
REL="RAG/ragflow"
FULL="$ROOT/$REL"

if [[ -e "$FULL" ]]; then
  echo "[init-submodule] Path already exists: $FULL — remove it (or drop submodule use) before adding." >&2
  exit 1
fi

echo "[init-submodule] Adding submodule $REL from $ROOT"
cd "$ROOT"
git submodule add https://github.com/infiniflow/ragflow.git "$REL"
git submodule update --init --recursive
echo "[init-submodule] Done."
echo "Edit RAG/.gitignore: remove the 'ragflow/' line so the submodule is not ignored."
echo "Then: git add RAG/.gitignore .gitmodules RAG/ragflow && git commit -m 'Add ragflow submodule'"
