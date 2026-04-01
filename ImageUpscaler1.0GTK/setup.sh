#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"
ARCH="$(uname -m)"

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
  log "Installing dependencies with Homebrew... (os=${OS}, arch=${ARCH})"
  brew update
  brew install cmake pkg-config gtk4 onnxruntime curl
}

install_apt() {
  need_cmd sudo
  need_cmd apt-get
  log "Installing dependencies with apt... (os=${OS}, arch=${ARCH})"
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgtk-4-dev \
    libcurl4-openssl-dev

  # libonnxruntime-dev is available on many distros/architectures but not all.
  if apt-cache show libonnxruntime-dev >/dev/null 2>&1; then
    sudo apt-get install -y libonnxruntime-dev
  else
    warn "apt repo has no libonnxruntime-dev for this arch (${ARCH})."
    warn "Install ONNX Runtime manually and set ONNXRUNTIME_ROOT."
  fi
}

install_dnf() {
  need_cmd sudo
  need_cmd dnf
  log "Installing dependencies with dnf... (os=${OS}, arch=${ARCH})"
  sudo dnf install -y \
    gcc-c++ \
    cmake \
    pkgconf-pkg-config \
    gtk4-devel \
    libcurl-devel
  if dnf list available onnxruntime-devel >/dev/null 2>&1; then
    sudo dnf install -y onnxruntime-devel
  else
    warn "dnf repo has no onnxruntime-devel for this arch (${ARCH})."
    warn "Install ONNX Runtime manually and set ONNXRUNTIME_ROOT."
  fi
}

install_pacman() {
  need_cmd sudo
  need_cmd pacman
  log "Installing dependencies with pacman... (os=${OS}, arch=${ARCH})"
  sudo pacman -Sy --needed --noconfirm \
    base-devel \
    cmake \
    pkgconf \
    gtk4 \
    curl
  if pacman -Ss '^onnxruntime$' >/dev/null 2>&1; then
    sudo pacman -Sy --needed --noconfirm onnxruntime
  else
    warn "pacman repo has no onnxruntime package for this arch (${ARCH})."
    warn "Install ONNX Runtime manually and set ONNXRUNTIME_ROOT."
  fi
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
    "/opt/homebrew/opt/onnxruntime/include/onnxruntime/onnxruntime_cxx_api.h"
    "/usr/local/opt/onnxruntime/include/onnxruntime/onnxruntime_cxx_api.h"
    "/usr/include/onnxruntime/onnxruntime_cxx_api.h"
    "/usr/include/onnxruntime_cxx_api.h"
  )
  local lib_candidates=(
    "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib"
    "/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib"
    "/usr/lib/x86_64-linux-gnu/libonnxruntime.so"
    "/usr/lib/aarch64-linux-gnu/libonnxruntime.so"
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
    warn "If build fails, install ONNX Runtime manually and set ONNXRUNTIME_ROOT:"
    warn "  export ONNXRUNTIME_ROOT=/path/to/onnxruntime"
    warn "  # expected: \$ONNXRUNTIME_ROOT/include/onnxruntime/*.h and lib/libonnxruntime.*"
  fi
}

print_next_steps() {
  cat <<EOF

Done.

Next steps:
  make -C "$PROJECT_DIR" build
  make -C "$PROJECT_DIR" run

EOF
}

main() {
  case "$OS" in
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
      err "Unsupported OS: $OS (arch=${ARCH})"
      exit 1
      ;;
  esac

  verify_deps
  print_next_steps
}

main "$@"
