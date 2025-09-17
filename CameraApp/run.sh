#!/bin/bash

# CameraApp Run Script
# This script provides easy access to the webcam application from the root directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/Source"

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    echo "CameraApp - Real-time Face Detection"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deps         Install system dependencies only"
    echo "  build        Build the application (auto-installs dependencies)"
    echo "  build-static Build with static linking (portable executable)"
    echo "  run          Build and run the application"
    echo "  test         Run tests"
    echo "  clean        Clean build files"
    echo "  info         Show build information"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deps         # Install system dependencies"
    echo "  $0 build        # Build the application"
    echo "  $0 build-static # Build with static linking"
    echo "  $0 run          # Build and run the application"
    echo "  $0 clean        # Clean build files"
    echo ""
    echo "For more detailed options, see: $SOURCE_DIR/build_webcam.sh --help"
}

# Check if Source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Check if build script exists
if [ ! -f "$SOURCE_DIR/build_webcam.sh" ]; then
    print_error "Build script not found: $SOURCE_DIR/build_webcam.sh"
    exit 1
fi

# Make sure the build script is executable
chmod +x "$SOURCE_DIR/build_webcam.sh"

# Main function
main() {
    case "${1:-help}" in
        "deps"|"build"|"build-static"|"run"|"test"|"clean"|"info")
            print_status "Executing: $1"
            cd "$SOURCE_DIR"
            ./build_webcam.sh "$1"
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
