#!/usr/bin/env bash
# Merge RAG/config/env.override into ragflow/docker/.env
set -euo pipefail
RAG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OVERRIDE="$RAG_DIR/config/env.override"
ENV_FILE="$RAG_DIR/ragflow/docker/.env"

if [[ ! -f "$OVERRIDE" ]]; then
  echo "[apply-overrides] No config/env.override — copy from config/env.override.example"
  exit 0
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — run bootstrap.sh first." >&2
  exit 1
fi

cp "$ENV_FILE" "$ENV_FILE.bak"
python3 - "$OVERRIDE" "$ENV_FILE" <<'PY'
import sys
from pathlib import Path

override_path, env_path = Path(sys.argv[1]), Path(sys.argv[2])
overrides = {}
for line in override_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    k, v = k.strip(), v.strip()
    if k:
        overrides[k] = v

if not overrides:
    print("[apply-overrides] No key=value entries in env.override")
    sys.exit(0)

lines = env_path.read_text(encoding="utf-8").splitlines()
seen = set()
out = []
for ln in lines:
    stripped = ln.strip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        k = stripped.split("=", 1)[0].strip()
        if k in overrides:
            out.append(f"{k}={overrides[k]}")
            seen.add(k)
            continue
    out.append(ln)
for k, v in overrides.items():
    if k not in seen:
        out.append(f"{k}={v}")

env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"[apply-overrides] Updated {env_path} ({len(overrides)} keys). Backup: .env.bak")
PY
