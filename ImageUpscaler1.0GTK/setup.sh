#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Required command not found: $1"
    exit 1
  fi
}

install_macos() {
  need_cmd brew
  log "Installing dependencies with Homebrew..."
  brew update
  brew install cmake pkg-config gtk4 onnxruntime curl
}

install_apt() {
  need_cmd sudo
  need_cmd apt-get
  log "Installing dependencies with apt..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgtk-4-dev \
    libcurl4-openssl-dev \
    libonnxruntime-dev
}

install_dnf() {
  need_cmd sudo
  need_cmd dnf
  log "Installing dependencies with dnf..."
  sudo dnf install -y \
    gcc-c++ \
    cmake \
    pkgconf-pkg-config \
    gtk4-devel \
    libcurl-devel \
    onnxruntime-devel || true
}

install_pacman() {
  need_cmd sudo
  need_cmd pacman
  log "Installing dependencies with pacman..."
  sudo pacman -Sy --needed --noconfirm \
    base-devel \
    cmake \
    pkgconf \
    gtk4 \
    curl \
    onnxruntime
}

verify_deps() {
  log "Verifying core tools..."
  need_cmd cmake
  need_cmd pkg-config

  if ! pkg-config --exists gtk4; then
    err "gtk4 pkg-config entry not found."
    exit 1
  fi

  local has_ort_header=0
  local has_ort_lib=0
  local header_candidates=(
    "/opt/homebrew/opt/onnxruntime/include/onnxruntime_cxx_api.h"
    "/usr/local/opt/onnxruntime/include/onnxruntime_cxx_api.h"
    "/usr/include/onnxruntime/onnxruntime_cxx_api.h"
    "/usr/include/onnxruntime_cxx_api.h"
  )
  local lib_candidates=(
    "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib"
    "/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib"
    "/usr/lib/x86_64-linux-gnu/libonnxruntime.so"
    "/usr/lib/libonnxruntime.so"
  )

  for p in "${header_candidates[@]}"; do
    if [[ -f "$p" ]]; then
      has_ort_header=1
      break
    fi
  done
  for p in "${lib_candidates[@]}"; do
    if [[ -f "$p" ]]; then
      has_ort_lib=1
      break
    fi
  done

  if [[ "${has_ort_header}" -eq 0 || "${has_ort_lib}" -eq 0 ]]; then
    warn "ONNX Runtime header/lib was not found in default paths."
    warn "If CMake fails, set ONNXRUNTIME_ROOT manually, for example:"
    warn "  export ONNXRUNTIME_ROOT=/path/to/onnxruntime"
  fi
}

print_next_steps() {
  cat <<EOF

Done.

Next steps:
  cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/build"
  cmake --build "$PROJECT_DIR/build" -j
  "$PROJECT_DIR/build/image_upscaler_gtk"

EOF
}

main() {
  local os
  os="$(uname -s)"

  case "$os" in
    Darwin)
      install_macos
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        install_apt
      elif command -v dnf >/dev/null 2>&1; then
        install_dnf
      elif command -v pacman >/dev/null 2>&1; then
        install_pacman
      else
        err "Unsupported Linux package manager. Supported: apt, dnf, pacman."
        exit 1
      fi
      ;;
    *)
      err "Unsupported OS: $os"
      exit 1
      ;;
  esac

  verify_deps
  print_next_steps
}

main "$@"
