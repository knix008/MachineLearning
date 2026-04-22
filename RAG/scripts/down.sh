#!/usr/bin/env bash
set -euo pipefail
RAG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCKER_DIR="$RAG_DIR/ragflow/docker"

if [[ ! -f "$DOCKER_DIR/docker-compose.yml" ]]; then
  echo "ragflow/docker not found." >&2
  exit 1
fi

cd "$DOCKER_DIR"
docker compose -f docker-compose.yml down
