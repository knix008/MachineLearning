#!/bin/bash

# Static Build Script for CameraApp
# This script builds the application with static linking

set -e  # Exit on any error

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

# Function to check if static libraries are available
check_static_libraries() {
    print_status "Checking for static libraries..."
    
    local missing_libs=()
    
    # Check system static libraries
    if [ ! -f "/usr/lib/x86_64-linux-gnu/libgtk-3.a" ]; then
        missing_libs+=("libgtk-3-dev (static)")
    fi
    
    if [ ! -f "/usr/lib/x86_64-linux-gnu/libglib-2.0.a" ]; then
        missing_libs+=("libglib2.0-dev (static)")
    fi
    
    if [ ! -f "/usr/lib/x86_64-linux-gnu/libcairo.a" ]; then
        missing_libs+=("libcairo2-dev (static)")
    fi
    
    # Check OpenCV static libraries
    if [ ! -f "opencv/lib/libopencv_core.a" ]; then
        missing_libs+=("OpenCV static libraries")
    fi
    
    # Check ONNX Runtime static libraries
    if [ ! -f "onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.a" ]; then
        missing_libs+=("ONNX Runtime static libraries")
    fi
    
    if [ ${#missing_libs[@]} -ne 0 ]; then
        print_error "Missing static libraries:"
        for lib in "${missing_libs[@]}"; do
            echo "  - $lib"
        done
        echo
        print_warning "To install static libraries, run:"
        echo "  sudo apt install -y libgtk-3-dev libglib2.0-dev libcairo2-dev libpango1.0-dev libatk1.0-dev libgdk-pixbuf2.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev"
        echo
        print_warning "For OpenCV static libraries, you need to rebuild OpenCV with static linking."
        echo "For ONNX Runtime static libraries, you need to download the static version."
        echo
        print_status "Would you like to continue with dynamic linking instead? (y/N)"
        read -p "Continue with dynamic build? " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Static build cancelled."
            exit 1
        fi
        return 1
    fi
    
    print_success "All static libraries found"
    return 0
}

# Function to build OpenCV with static libraries
build_opencv_static() {
    print_status "Building OpenCV with static libraries..."
    
    if [ ! -f "build_opencv.sh" ]; then
        print_error "build_opencv.sh not found"
        exit 1
    fi
    
    # Modify OpenCV build to use static libraries
    print_status "Configuring OpenCV for static build..."
    
    # This would require modifying the OpenCV build script
    # For now, we'll use the existing dynamic OpenCV
    print_warning "OpenCV static build not implemented yet. Using dynamic OpenCV."
}

# Function to build the application with static linking
build_static_application() {
    print_status "Building application with static linking..."
    
    # Clean previous build
    if [ -d "build" ]; then
        print_status "Cleaning previous build..."
        rm -rf build
    fi
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with static linking
    print_status "Configuring CMake with static linking..."
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON ..
    
    # Build
    print_status "Building application..."
    make -j$(nproc)
    
    print_success "Static build completed successfully"
    
    # Check if the executable is truly static
    print_status "Checking executable dependencies..."
    if command -v ldd &> /dev/null; then
        echo "Dynamic dependencies:"
        ldd webcam_app | head -10
        echo "..."
    fi
    
    # Check file size
    local size=$(du -h webcam_app | cut -f1)
    print_status "Executable size: $size"
}

# Function to show help
show_help() {
    echo "Static Build Script for CameraApp"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -f, --force    Force build even if static libraries are missing"
    echo "  --check        Only check for static libraries"
    echo
    echo "This script will:"
    echo "  1. Check for required static libraries"
    echo "  2. Build the application with static linking"
    echo "  3. Create a standalone executable"
    echo
    echo "Requirements:"
    echo "  - Static development libraries installed"
    echo "  - OpenCV built with static libraries"
    echo "  - ONNX Runtime static libraries"
}

# Parse command line arguments
FORCE_BUILD=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE_BUILD=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "CameraApp Static Build Script"
    print_status "=============================="
    
    # Check for static libraries
    if ! check_static_libraries; then
        if [ "$FORCE_BUILD" = false ]; then
            print_error "Cannot proceed with static build due to missing libraries"
            exit 1
        else
            print_warning "Proceeding with force flag despite missing libraries"
        fi
    fi
    
    if [ "$CHECK_ONLY" = true ]; then
        print_success "Library check completed"
        exit 0
    fi
    
    # Build the application
    build_static_application
    
    print_success "Static build completed!"
    print_status "Executable location: build/webcam_app"
    print_status "You can now run: ./build/webcam_app"
}

# Run main function
main "$@"
