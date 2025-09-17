#!/bin/bash

# CameraApp Dependency Installation Script
# This script automatically installs all required system dependencies
# for building and running the CameraApp

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

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    elif [ -f /etc/redhat-release ]; then
        echo "rhel"
    elif [ -f /etc/arch-release ]; then
        echo "arch"
    else
        echo "unknown"
    fi
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "This script should not be run as root (sudo)."
        print_error "It will prompt for sudo when needed."
        exit 1
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if package is installed (Ubuntu/Debian)
package_installed_apt() {
    dpkg -l | grep -q "^ii  $1 " 2>/dev/null
}

# Function to check if package is installed (CentOS/RHEL/Fedora)
package_installed_yum() {
    rpm -q "$1" >/dev/null 2>&1
}

# Function to check if package is installed (Arch)
package_installed_pacman() {
    pacman -Q "$1" >/dev/null 2>&1
}

# Function to install packages on Ubuntu/Debian
install_packages_apt() {
    local packages=("$@")
    local missing_packages=()
    
    print_status "Checking packages on Ubuntu/Debian..."
    
    for package in "${packages[@]}"; do
        if ! package_installed_apt "$package"; then
            missing_packages+=("$package")
        else
            print_success "✓ $package is already installed"
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_status "Installing missing packages: ${missing_packages[*]}"
        sudo apt update
        sudo apt install -y "${missing_packages[@]}"
        print_success "Package installation completed"
    else
        print_success "All required packages are already installed"
    fi
}

# Function to install packages on CentOS/RHEL/Fedora
install_packages_yum() {
    local packages=("$@")
    local missing_packages=()
    
    print_status "Checking packages on CentOS/RHEL/Fedora..."
    
    for package in "${packages[@]}"; do
        if ! package_installed_yum "$package"; then
            missing_packages+=("$package")
        else
            print_success "✓ $package is already installed"
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_status "Installing missing packages: ${missing_packages[*]}"
        sudo yum install -y "${missing_packages[@]}"
        print_success "Package installation completed"
    else
        print_success "All required packages are already installed"
    fi
}

# Function to install packages on Arch Linux
install_packages_pacman() {
    local packages=("$@")
    local missing_packages=()
    
    print_status "Checking packages on Arch Linux..."
    
    for package in "${packages[@]}"; do
        if ! package_installed_pacman "$package"; then
            missing_packages+=("$package")
        else
            print_success "✓ $package is already installed"
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_status "Installing missing packages: ${missing_packages[*]}"
        sudo pacman -S --noconfirm "${missing_packages[@]}"
        print_success "Package installation completed"
    else
        print_success "All required packages are already installed"
    fi
}

# Function to check and install build tools
install_build_tools() {
    local distro="$1"
    
    print_status "Installing build tools..."
    
    case "$distro" in
        "ubuntu"|"debian")
            install_packages_apt \
                cmake \
                make \
                build-essential \
                git \
                wget \
                tar \
                pkg-config
            ;;
        "rhel"|"centos"|"fedora")
            install_packages_yum \
                cmake \
                make \
                gcc-c++ \
                git \
                wget \
                tar \
                pkgconfig
            ;;
        "arch")
            install_packages_pacman \
                cmake \
                make \
                gcc \
                git \
                wget \
                tar \
                pkg-config
            ;;
        *)
            print_error "Unsupported distribution: $distro"
            exit 1
            ;;
    esac
}

# Function to check and install GStreamer development packages
install_gstreamer_dev() {
    local distro="$1"
    
    print_status "Installing GStreamer development packages..."
    
    case "$distro" in
        "ubuntu"|"debian")
            install_packages_apt \
                libgstreamer1.0-dev \
                libgstreamer-plugins-base1.0-dev
            ;;
        "rhel"|"centos"|"fedora")
            install_packages_yum \
                gstreamer1-devel \
                gstreamer1-plugins-base-devel
            ;;
        "arch")
            install_packages_pacman \
                gstreamer \
                gst-plugins-base
            ;;
        *)
            print_error "Unsupported distribution: $distro"
            exit 1
            ;;
    esac
}

# Function to check and install GTK+3 development packages
install_gtk3_dev() {
    local distro="$1"
    
    print_status "Installing GTK+3 development packages..."
    
    case "$distro" in
        "ubuntu"|"debian")
            install_packages_apt \
                libgtk-3-dev \
                libglib2.0-dev \
                libcairo2-dev \
                libpango1.0-dev \
                libatk1.0-dev \
                libgdk-pixbuf2.0-dev
            ;;
        "rhel"|"centos"|"fedora")
            install_packages_yum \
                gtk3-devel \
                glib2-devel \
                cairo-devel \
                pango-devel \
                atk-devel \
                gdk-pixbuf2-devel
            ;;
        "arch")
            install_packages_pacman \
                gtk3 \
                glib2 \
                cairo \
                pango \
                atk \
                gdk-pixbuf2
            ;;
        *)
            print_error "Unsupported distribution: $distro"
            exit 1
            ;;
    esac
}

# Function to verify installations
verify_installations() {
    print_status "Verifying installations..."
    
    # Check build tools
    local missing_tools=()
    
    if ! command_exists cmake; then
        missing_tools+=("cmake")
    fi
    
    if ! command_exists make; then
        missing_tools+=("make")
    fi
    
    if ! command_exists gcc; then
        missing_tools+=("gcc")
    fi
    
    if ! command_exists git; then
        missing_tools+=("git")
    fi
    
    if ! command_exists wget; then
        missing_tools+=("wget")
    fi
    
    if ! command_exists pkg-config; then
        missing_tools+=("pkg-config")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing build tools: ${missing_tools[*]}"
        return 1
    fi
    
    # Check GStreamer
    if ! pkg-config --exists gstreamer-1.0; then
        print_error "GStreamer development packages not found"
        return 1
    fi
    
    # Check GTK+3
    if ! pkg-config --exists gtk+-3.0; then
        print_error "GTK+3 development packages not found"
        return 1
    fi
    
    print_success "All dependencies verified successfully"
    return 0
}

# Function to show installation summary
show_summary() {
    print_success "Dependency installation completed!"
    echo
    print_status "Installed packages:"
    echo "  ✓ Build tools: cmake, make, gcc, git, wget, pkg-config"
    echo "  ✓ GStreamer development: libgstreamer1.0-dev, libgstreamer-plugins-base1.0-dev"
    echo "  ✓ GTK+3 development: libgtk-3-dev, libglib2.0-dev, libcairo2-dev, libpango1.0-dev, libatk1.0-dev, libgdk-pixbuf2.0-dev"
    echo
    print_status "Next steps:"
    echo "  1. Run: ./build_webcam.sh build"
    echo "  2. Or run: ./run.sh build"
    echo
}

# Function to show help
show_help() {
    echo "CameraApp Dependency Installation Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --version  Show version information"
    echo "  -f, --force    Force reinstall all packages"
    echo "  --build-only   Install only build tools"
    echo "  --gstreamer    Install only GStreamer packages"
    echo "  --gtk3         Install only GTK+3 packages"
    echo
    echo "This script will automatically:"
    echo "  1. Detect your Linux distribution"
    echo "  2. Install required build tools"
    echo "  3. Install GStreamer development packages"
    echo "  4. Install GTK+3 development packages"
    echo "  5. Verify all installations"
    echo
    echo "Supported distributions:"
    echo "  - Ubuntu/Debian (apt)"
    echo "  - CentOS/RHEL/Fedora (yum)"
    echo "  - Arch Linux (pacman)"
    echo
    echo "The script will prompt for sudo when needed."
}

# Function to show version
show_version() {
    echo "CameraApp Dependency Installer v1.0"
    echo "Target: CameraApp with OpenCV 4.8.1 and ONNX Runtime 1.16.3"
}

# Parse command line arguments
FORCE_INSTALL=false
BUILD_ONLY=false
GSTREAMER_ONLY=false
GTK3_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
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
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --gstreamer)
            GSTREAMER_ONLY=true
            shift
            ;;
        --gtk3)
            GTK3_ONLY=true
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
    print_status "CameraApp Dependency Installation Script"
    print_status "=========================================="
    
    # Check if not running as root
    check_root
    
    # Detect distribution
    local distro=$(detect_distro)
    print_status "Detected distribution: $distro"
    
    # Install dependencies based on options
    if [ "$BUILD_ONLY" = true ]; then
        install_build_tools "$distro"
    elif [ "$GSTREAMER_ONLY" = true ]; then
        install_gstreamer_dev "$distro"
    elif [ "$GTK3_ONLY" = true ]; then
        install_gtk3_dev "$distro"
    else
        # Install all dependencies
        install_build_tools "$distro"
        install_gstreamer_dev "$distro"
        install_gtk3_dev "$distro"
    fi
    
    # Verify installations
    if verify_installations; then
        show_summary
    else
        print_error "Dependency verification failed"
        exit 1
    fi
}

# Run main function
main "$@"
