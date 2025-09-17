#!/bin/bash

# ONNX Runtime Installation Script
# This script downloads and installs ONNX Runtime 1.16.3 for the webcam application

set -e  # Exit on any error

# Configuration
ONNX_VERSION="1.16.3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to detect CPU architecture
detect_architecture() {
    local arch
    arch=$(uname -m)
    
    case "$arch" in
        x86_64)
            echo "x64"
            ;;
        aarch64|arm64)
            echo "aarch64"
            ;;
        armv7l|armv8l)
            echo "aarch64"
            ;;
        *)
            print_error "Unsupported architecture: $arch"
            print_error "Supported architectures: x86_64 (x64), aarch64, arm64"
            exit 1
            ;;
    esac
}

# Detect architecture and set URLs
ARCHITECTURE=$(detect_architecture)
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-${ARCHITECTURE}-${ONNX_VERSION}.tgz"
ONNX_ARCHIVE="onnxruntime-linux-${ARCHITECTURE}-${ONNX_VERSION}.tgz"
ONNX_DIR="onnxruntime-linux-${ARCHITECTURE}-${ONNX_VERSION}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if directory exists and is not empty
directory_exists_and_not_empty() {
    [ -d "$1" ] && [ "$(ls -A "$1" 2>/dev/null)" ]
}

# Function to cleanup on exit
cleanup() {
    if [ -f "$ONNX_ARCHIVE" ]; then
        print_status "Cleaning up temporary files..."
        rm -f "$ONNX_ARCHIVE"
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Main installation function
install_onnxruntime() {
    print_status "Starting ONNX Runtime ${ONNX_VERSION} installation..."
    print_status "Detected architecture: ${ARCHITECTURE}"
    
    # Check if we're in the right directory
    if [ ! -f "CMakeLists.txt" ]; then
        print_error "CMakeLists.txt not found. Please run this script from the Source directory."
        exit 1
    fi
    
    # Check if ONNX Runtime is already installed
    if directory_exists_and_not_empty "$ONNX_DIR"; then
        print_warning "ONNX Runtime directory already exists: $ONNX_DIR"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Installation cancelled."
            exit 0
        fi
        print_status "Removing existing installation..."
        rm -rf "$ONNX_DIR"
    fi
    
    # Check for required tools
    print_status "Checking required tools..."
    
    if ! command_exists wget; then
        print_error "wget is not installed. Please install it first:"
        print_error "  sudo apt update && sudo apt install wget"
        exit 1
    fi
    
    if ! command_exists tar; then
        print_error "tar is not installed. Please install it first:"
        print_error "  sudo apt update && sudo apt install tar"
        exit 1
    fi
    
    # Download ONNX Runtime
    print_status "Downloading ONNX Runtime ${ONNX_VERSION}..."
    if wget -q --show-progress "$ONNX_URL" -O "$ONNX_ARCHIVE"; then
        print_success "Download completed successfully"
    else
        print_error "Failed to download ONNX Runtime"
        exit 1
    fi
    
    # Verify download
    if [ ! -f "$ONNX_ARCHIVE" ]; then
        print_error "Downloaded file not found"
        exit 1
    fi
    
    # Extract ONNX Runtime
    print_status "Extracting ONNX Runtime..."
    if tar -xzf "$ONNX_ARCHIVE"; then
        print_success "Extraction completed successfully"
    else
        print_error "Failed to extract ONNX Runtime"
        exit 1
    fi
    
    # Verify extraction
    if [ ! -d "$ONNX_DIR" ]; then
        print_error "Extracted directory not found"
        exit 1
    fi
    
    # Check for required files
    print_status "Verifying installation..."
    
    if [ ! -f "$ONNX_DIR/lib/libonnxruntime.so" ]; then
        print_error "ONNX Runtime library not found"
        exit 1
    fi
    
    if [ ! -f "$ONNX_DIR/include/onnxruntime_c_api.h" ]; then
        print_error "ONNX Runtime headers not found"
        exit 1
    fi
    
    # Set permissions
    print_status "Setting permissions..."
    chmod -R 755 "$ONNX_DIR"
    
    # Display installation info
    print_success "ONNX Runtime ${ONNX_VERSION} installed successfully!"
    echo
    print_status "Installation details:"
    echo "  Directory: $ONNX_DIR"
    echo "  Library: $ONNX_DIR/lib/libonnxruntime.so"
    echo "  Headers: $ONNX_DIR/include/"
    echo "  Size: $(du -sh "$ONNX_DIR" | cut -f1)"
    echo
    
    # Test CMake configuration
    print_status "Testing CMake configuration..."
    if [ -d "build" ]; then
        cd build
        if cmake .. >/dev/null 2>&1; then
            print_success "CMake configuration successful"
        else
            print_warning "CMake configuration failed - you may need to run 'make clean' first"
        fi
        cd ..
    else
        print_warning "Build directory not found - run 'mkdir build && cd build && cmake ..' to test"
    fi
    
    echo
    print_success "Installation completed! You can now build your webcam application."
    print_status "To build: cd build && make"
    print_status "To run: make run-webcam"
}

# Function to show usage
show_usage() {
    echo "ONNX Runtime Installation Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --version  Show version information"
    echo "  -f, --force    Force reinstallation even if already installed"
    echo
    echo "This script will:"
echo "  1. Detect your CPU architecture (x64 or aarch64)"
echo "  2. Download ONNX Runtime ${ONNX_VERSION} for your architecture"
echo "  3. Extract it to the current directory"
echo "  4. Verify the installation"
echo "  5. Test CMake configuration"
    echo
    echo "Requirements:"
    echo "  - wget (for downloading)"
    echo "  - tar (for extraction)"
    echo "  - Internet connection"
    echo
    echo "The script must be run from the Source directory containing CMakeLists.txt"
}

# Function to show version
show_version() {
    echo "ONNX Runtime Installation Script v1.0"
    echo "Target ONNX Runtime version: ${ONNX_VERSION}"
    echo "Supported platforms: Linux x64, Linux aarch64"
    echo "Detected architecture: ${ARCHITECTURE}"
}

# Parse command line arguments
FORCE_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--version)
            show_version
            exit 0
            ;;
        -f|--force)
            FORCE_INSTALL=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
if [ "$FORCE_INSTALL" = true ]; then
    if directory_exists_and_not_empty "$ONNX_DIR"; then
        print_status "Force flag detected - removing existing installation..."
        rm -rf "$ONNX_DIR"
    fi
fi

install_onnxruntime
