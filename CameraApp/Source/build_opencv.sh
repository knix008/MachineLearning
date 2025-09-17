#!/bin/bash

# OpenCV Build Script for Webcam Application
# This script builds OpenCV 4.8.1 locally with optimized settings

set -e  # Exit on any error

# Configuration
OPENCV_VERSION="4.8.1"
OPENCV_SOURCE_DIR="opencv_src"
OPENCV_BUILD_DIR="opencv_build"
OPENCV_INSTALL_DIR="opencv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    print_status "Cleaning up temporary files..."
    if [ -d "$OPENCV_BUILD_DIR" ]; then
        rm -rf "$OPENCV_BUILD_DIR"
    fi
    if [ -d "$OPENCV_SOURCE_DIR" ]; then
        rm -rf "$OPENCV_SOURCE_DIR"
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Main build function
build_opencv() {
    print_status "Starting OpenCV ${OPENCV_VERSION} build..."
    print_status "Detected architecture: ${ARCHITECTURE}"
    
    # Check if we're in the right directory
    if [ ! -f "CMakeLists.txt" ]; then
        print_error "CMakeLists.txt not found. Please run this script from the Source directory."
        exit 1
    fi
    
    # Check if OpenCV is already built
    if directory_exists_and_not_empty "$OPENCV_INSTALL_DIR"; then
        print_warning "OpenCV directory already exists: $OPENCV_INSTALL_DIR"
        read -p "Do you want to rebuild? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Build cancelled."
            exit 0
        fi
        print_status "Removing existing installation..."
        rm -rf "$OPENCV_INSTALL_DIR"
    fi
    
    # Check for required tools
    print_status "Checking required tools..."
    
    local missing_tools=()
    
    if ! command_exists cmake; then
        missing_tools+=("cmake")
    fi
    
    if ! command_exists make; then
        missing_tools+=("make")
    fi
    
    if ! command_exists git; then
        missing_tools+=("git")
    fi
    
    if ! command_exists wget; then
        missing_tools+=("wget")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_error "Please install them first:"
        print_error "  sudo apt update && sudo apt install ${missing_tools[*]}"
        exit 1
    fi
    
    # Check for system dependencies
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check for development packages (only essential ones that OpenCV can't build itself)
    if ! pkg-config --exists gstreamer-1.0; then
        missing_deps+=("libgstreamer1.0-dev")
    fi
    
    if ! pkg-config --exists gstreamer-base-1.0; then
        missing_deps+=("libgstreamer-plugins-base1.0-dev")
    fi
    
    # Note: OpenCV builds its own versions of:
    # - Image codecs: JPEG, PNG, TIFF, WebP, Zlib
    # - Video codecs: FFmpeg (libavcodec, libavformat, libavutil, libswscale)
    # so we don't need to check for system packages for these
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing system dependencies: ${missing_deps[*]}"
        print_error "Please install them first:"
        print_error "  sudo apt update && sudo apt install ${missing_deps[*]}"
        exit 1
    fi
    
    print_success "All dependencies found"
    
    # Download OpenCV source
    print_status "Downloading OpenCV ${OPENCV_VERSION} source..."
    
    if [ ! -d "$OPENCV_SOURCE_DIR" ]; then
        mkdir -p "$OPENCV_SOURCE_DIR"
        cd "$OPENCV_SOURCE_DIR"
        
        # Download OpenCV
        wget -q --show-progress "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz" -O "opencv-${OPENCV_VERSION}.tar.gz"
        wget -q --show-progress "https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz" -O "opencv_contrib-${OPENCV_VERSION}.tar.gz"
        
        # Extract
        tar -xzf "opencv-${OPENCV_VERSION}.tar.gz"
        tar -xzf "opencv_contrib-${OPENCV_VERSION}.tar.gz"
        
        # Rename directories
        mv "opencv-${OPENCV_VERSION}" "opencv"
        mv "opencv_contrib-${OPENCV_VERSION}" "opencv_contrib"
        
        cd ..
    else
        print_status "OpenCV source already exists, skipping download"
    fi
    
    # Create build directory
    print_status "Creating build directory..."
    mkdir -p "$OPENCV_BUILD_DIR"
    cd "$OPENCV_BUILD_DIR"
    
    # Configure CMake
    print_status "Configuring OpenCV with CMake..."
    
    cmake ../$OPENCV_SOURCE_DIR/opencv \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR/$OPENCV_INSTALL_DIR" \
        -DOPENCV_EXTRA_MODULES_PATH="$SCRIPT_DIR/$OPENCV_SOURCE_DIR/opencv_contrib/modules" \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_java=OFF \
        -DBUILD_opencv_python=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_js=OFF \
        -DBUILD_opencv_ts=OFF \
        -DBUILD_opencv_world=OFF \
        -DWITH_1394=OFF \
        -DWITH_CAROTENE=OFF \
        -DWITH_CUBLAS=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_CUFFT=OFF \
        -DWITH_CURAND=OFF \
        -DWITH_CUSOLVER=OFF \
        -DWITH_CUSPARSE=OFF \
        -DWITH_EIGEN=ON \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=ON \
        -DWITH_GTK=ON \
        -DWITH_IPP=OFF \
        -DWITH_ITT=OFF \
        -DWITH_JASPER=OFF \
        -DWITH_JPEG=ON \
        -DWITH_LAPACK=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENCLAMDBLAS=OFF \
        -DWITH_OPENCLAMDFFT=OFF \
        -DWITH_OPENCL_SVM=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_OPENGL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_OPENNI=OFF \
        -DWITH_OPENNI2=OFF \
        -DWITH_OPENVX=OFF \
        -DWITH_PNG=ON \
        -DWITH_PROTOBUF=OFF \
        -DWITH_PTHREADS_PF=ON \
        -DWITH_PVAPI=OFF \
        -DWITH_QT=OFF \
        -DWITH_TBB=OFF \
        -DWITH_TIFF=ON \
        -DWITH_UNICAP=OFF \
        -DWITH_V4L=ON \
        -DWITH_VA=OFF \
        -DWITH_VA_INTEL=OFF \
        -DWITH_VDPAU=OFF \
        -DWITH_VTK=OFF \
        -DWITH_WEBP=ON \
        -DWITH_XIMEA=OFF \
        -DWITH_XINE=OFF \
        -DWITH_XVID=OFF \
        -DWITH_ZLIB=ON \
        -DBUILD_JPEG=ON \
        -DBUILD_PNG=ON \
        -DBUILD_TIFF=ON \
        -DBUILD_WEBP=ON \
        -DBUILD_ZLIB=ON \
        -DBUILD_FFMPEG=ON \
        -DOPENCV_ENABLE_NONFREE=OFF \
        -DOPENCV_GENERATE_PKGCONFIG=OFF \
        -DOPENCV_PYTHON_INSTALL_PATH="" \
        -DOPENCV_SKIP_PYTHON_LOADER=ON \
        -DOPENCV_SKIP_ANDROID_TESTS=ON \
        -DOPENCV_SKIP_GTK_INSTALL=ON \
        -DOPENCV_SKIP_QT_INSTALL=ON \
        -DOPENCV_SKIP_VTK_INSTALL=ON \
        -DOPENCV_SKIP_OPENCL_INSTALL=ON \
        -DOPENCV_SKIP_CUDA_INSTALL=ON \
        -DOPENCV_SKIP_IPP_INSTALL=ON \
        -DOPENCV_SKIP_ITT_INSTALL=ON \
        -DOPENCV_SKIP_TBB_INSTALL=ON \
        -DOPENCV_SKIP_OPENMP_INSTALL=ON \
        -DOPENCV_SKIP_EIGEN_INSTALL=ON \
        -DOPENCV_SKIP_LAPACK_INSTALL=ON \
        -DOPENCV_SKIP_GSTREAMER_INSTALL=ON \
        -DOPENCV_SKIP_FFMPEG_INSTALL=OFF \
        -DOPENCV_SKIP_JPEG_INSTALL=OFF \
        -DOPENCV_SKIP_PNG_INSTALL=OFF \
        -DOPENCV_SKIP_TIFF_INSTALL=OFF \
        -DOPENCV_SKIP_WEBP_INSTALL=OFF \
        -DOPENCV_SKIP_ZLIB_INSTALL=OFF \
        -DOPENCV_SKIP_1394_INSTALL=ON \
        -DOPENCV_SKIP_CAROTENE_INSTALL=ON \
        -DOPENCV_SKIP_CUBLAS_INSTALL=ON \
        -DOPENCV_SKIP_CUFFT_INSTALL=ON \
        -DOPENCV_SKIP_CURAND_INSTALL=ON \
        -DOPENCV_SKIP_CUSOLVER_INSTALL=ON \
        -DOPENCV_SKIP_CUSPARSE_INSTALL=ON \
        -DOPENCV_SKIP_IPP_INSTALL=ON \
        -DOPENCV_SKIP_ITT_INSTALL=ON \
        -DOPENCV_SKIP_JASPER_INSTALL=ON \
        -DOPENCV_SKIP_LAPACK_INSTALL=ON \
        -DOPENCV_SKIP_OPENCL_INSTALL=ON \
        -DOPENCV_SKIP_OPENCLAMDBLAS_INSTALL=ON \
        -DOPENCV_SKIP_OPENCLAMDFFT_INSTALL=ON \
        -DOPENCV_SKIP_OPENCL_SVM_INSTALL=ON \
        -DOPENCV_SKIP_OPENEXR_INSTALL=ON \
        -DOPENCV_SKIP_OPENGL_INSTALL=ON \
        -DOPENCV_SKIP_OPENMP_INSTALL=ON \
        -DOPENCV_SKIP_OPENNI_INSTALL=ON \
        -DOPENCV_SKIP_OPENNI2_INSTALL=ON \
        -DOPENCV_SKIP_OPENVX_INSTALL=ON \
        -DOPENCV_SKIP_PROTOBUF_INSTALL=ON \
        -DOPENCV_SKIP_PTHREADS_PF_INSTALL=ON \
        -DOPENCV_SKIP_PVAPI_INSTALL=ON \
        -DOPENCV_SKIP_QT_INSTALL=ON \
        -DOPENCV_SKIP_TBB_INSTALL=ON \
        -DOPENCV_SKIP_UNICAP_INSTALL=ON \
        -DOPENCV_SKIP_V4L_INSTALL=ON \
        -DOPENCV_SKIP_VA_INSTALL=ON \
        -DOPENCV_SKIP_VA_INTEL_INSTALL=ON \
        -DOPENCV_SKIP_VDPAU_INSTALL=ON \
        -DOPENCV_SKIP_VTK_INSTALL=ON \
        -DOPENCV_SKIP_XIMEA_INSTALL=ON \
        -DOPENCV_SKIP_XINE_INSTALL=ON \
        -DOPENCV_SKIP_XVID_INSTALL=ON \
        -DCMAKE_CXX_FLAGS="-O3 -march=native" \
        -DCMAKE_C_FLAGS="-O3 -march=native"
    
    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed"
        exit 1
    fi
    
    print_success "CMake configuration completed"
    
    # Build OpenCV
    print_status "Building OpenCV (this may take a while)..."
    
    local cpu_count=$(nproc)
    make -j$cpu_count
    
    if [ $? -ne 0 ]; then
        print_error "OpenCV build failed"
        exit 1
    fi
    
    print_success "OpenCV build completed"
    
    # Install OpenCV
    print_status "Installing OpenCV..."
    
    make install
    
    if [ $? -ne 0 ]; then
        print_error "OpenCV installation failed"
        exit 1
    fi
    
    cd ..
    
    # Verify installation
    print_status "Verifying installation..."
    
    if [ ! -f "$OPENCV_INSTALL_DIR/lib/libopencv_core.so" ]; then
        print_error "OpenCV installation verification failed"
        exit 1
    fi
    
    if [ ! -f "$OPENCV_INSTALL_DIR/include/opencv4/opencv2/opencv.hpp" ]; then
        print_error "OpenCV headers not found"
        exit 1
    fi
    
    # Set permissions
    print_status "Setting permissions..."
    chmod -R 755 "$OPENCV_INSTALL_DIR"
    
    # Display installation info
    print_success "OpenCV ${OPENCV_VERSION} built and installed successfully!"
    echo
    print_status "Installation details:"
    echo "  Directory: $OPENCV_INSTALL_DIR"
    echo "  Libraries: $OPENCV_INSTALL_DIR/lib/"
    echo "  Headers: $OPENCV_INSTALL_DIR/include/opencv4/"
    echo "  Size: $(du -sh "$OPENCV_INSTALL_DIR" | cut -f1)"
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
    print_success "OpenCV build completed! You can now build your webcam application."
    print_status "To build: cd build && make"
    print_status "To run: make run-webcam"
}

# Function to show usage
show_usage() {
    echo "OpenCV Build Script for Webcam Application"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --version  Show version information"
    echo "  -f, --force    Force rebuild even if already installed"
    echo "  -c, --clean    Clean build (remove existing installation)"
    echo
    echo "This script will:"
    echo "  1. Detect your CPU architecture (x64 or aarch64)"
    echo "  2. Check for required tools and dependencies"
    echo "  3. Download OpenCV ${OPENCV_VERSION} source code"
    echo "  4. Configure and build OpenCV with optimized settings"
    echo "  5. Install OpenCV to the local opencv directory"
    echo "  6. Test the installation with CMake"
    echo
    echo "Requirements:"
    echo "  - cmake (3.10+)"
    echo "  - make"
    echo "  - git"
    echo "  - wget"
    echo "  - System development libraries (see script for details)"
    echo "  - Internet connection"
    echo
    echo "The script must be run from the Source directory containing CMakeLists.txt"
}

# Function to show version
show_version() {
    echo "OpenCV Build Script v1.0"
    echo "Target OpenCV version: ${OPENCV_VERSION}"
    echo "Supported platforms: Linux x64, Linux aarch64"
    echo "Detected architecture: ${ARCHITECTURE}"
}

# Parse command line arguments
FORCE_BUILD=false
CLEAN_BUILD=false

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
            FORCE_BUILD=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Detect architecture
ARCHITECTURE=$(detect_architecture)

# Main execution
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Clean build requested - removing existing installation..."
    if directory_exists_and_not_empty "$OPENCV_INSTALL_DIR"; then
        rm -rf "$OPENCV_INSTALL_DIR"
        print_success "Existing installation removed"
    fi
fi

if [ "$FORCE_BUILD" = true ]; then
    if directory_exists_and_not_empty "$OPENCV_INSTALL_DIR"; then
        print_status "Force flag detected - removing existing installation..."
        rm -rf "$OPENCV_INSTALL_DIR"
    fi
fi

build_opencv
