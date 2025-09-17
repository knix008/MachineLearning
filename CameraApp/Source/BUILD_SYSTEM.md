# Build System Documentation

This document explains how to build the webcam application with local libraries.

## Overview

The project uses local versions of OpenCV and ONNX Runtime to ensure consistency and avoid system dependencies. The build system automatically detects your CPU architecture and configures everything accordingly.

## Prerequisites

### Required Tools
- `cmake` (3.10 or higher)
- `make`
- `git`
- `wget`
- `tar`

### System Dependencies
The following development packages are required for building OpenCV:

```bash
sudo apt update && sudo apt install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libatk1.0-dev \
    libgdk-pixbuf2.0-dev
```

**Note**: OpenCV builds its own versions of:
- **Image codecs**: JPEG, PNG, TIFF, WebP, Zlib
- **Video codecs**: FFmpeg (libavcodec, libavformat, libavutil, libswscale)

**GTK+3 dependencies** are required for OpenCV's GUI support (`-DWITH_GTK=ON`).
So system packages for these are not required.

**For comprehensive system dependency information, see [SYSTEM_DEPENDENCIES.md](SYSTEM_DEPENDENCIES.md)**

## Build Scripts

### 1. build_opencv.sh

Builds OpenCV 4.8.1 locally with optimized settings for your architecture.

**Usage:**
```bash
./build_opencv.sh [OPTIONS]
```

**Options:**
- `-h, --help`: Show help message
- `-v, --version`: Show version information
- `-f, --force`: Force rebuild even if already installed
- `-c, --clean`: Clean build (remove existing installation)

**Example:**
```bash
# First time build
./build_opencv.sh

# Force rebuild
./build_opencv.sh --force

# Clean build
./build_opencv.sh --clean
```

### 2. install_onnxruntime.sh

Downloads and installs ONNX Runtime 1.16.3 for your architecture.

**Usage:**
```bash
./install_onnxruntime.sh [OPTIONS]
```

**Options:**
- `-h, --help`: Show help message
- `-v, --version`: Show version information
- `-f, --force`: Force reinstallation even if already installed

**Example:**
```bash
# First time installation
./install_onnxruntime.sh

# Force reinstall
./install_onnxruntime.sh --force
```

## Build Process

### Step 1: Build OpenCV
```bash
cd Source
./build_opencv.sh
```

This will:
- Detect your CPU architecture (x64 or aarch64)
- Check for required tools and dependencies
- Download OpenCV 4.8.1 source code
- Configure and build OpenCV with optimized settings
- Install OpenCV to the local `opencv/` directory

### Step 2: Install ONNX Runtime
```bash
./install_onnxruntime.sh
```

This will:
- Detect your CPU architecture
- Download the appropriate ONNX Runtime version
- Extract it to the local directory
- Verify the installation

### Step 3: Build the Application
```bash
mkdir -p build
cd build
cmake ..
make
```

### Step 4: Run the Application
```bash
make run-webcam
```

## Architecture Support

The build system automatically detects and supports:
- **x64**: Intel/AMD 64-bit processors
- **aarch64**: ARM 64-bit processors (including Apple Silicon)

## Directory Structure

After building, your directory structure will look like:

```
Source/
├── opencv/                    # Local OpenCV installation
│   ├── lib/                  # OpenCV libraries
│   ├── include/              # OpenCV headers
│   └── share/                # OpenCV configuration files
├── onnxruntime-linux-x64-1.16.3/  # ONNX Runtime (architecture-specific)
│   ├── lib/                  # ONNX Runtime libraries
│   └── include/              # ONNX Runtime headers
├── build/                    # Build directory
│   ├── webcam_app           # Executable
│   └── models/              # Copied model files
├── models/                   # Model files
├── src/                      # Source code
├── include/                  # Header files
├── build_opencv.sh          # OpenCV build script
├── install_onnxruntime.sh   # ONNX Runtime installation script
└── CMakeLists.txt           # CMake configuration
```

## Troubleshooting

### OpenCV Build Issues

1. **Missing Dependencies**: If the build fails due to missing dependencies, install them:
   ```bash
   sudo apt update && sudo apt install [missing-package]
   ```
   
   **Note**: OpenCV builds its own versions of image codecs (JPEG, PNG, TIFF, WebP, Zlib), so you don't need to install system packages for these.

2. **Build Time**: OpenCV build can take 30-60 minutes depending on your system. Be patient.

3. **Memory Issues**: If you encounter memory issues during build, reduce the number of parallel jobs:
   ```bash
   # Edit build_opencv.sh and change:
   make -j$cpu_count
   # to:
   make -j2
   ```

### ONNX Runtime Issues

1. **Download Failures**: Check your internet connection and try again.

2. **Wrong Architecture**: The script automatically detects architecture, but if you have issues, check:
   ```bash
   uname -m
   ```

### CMake Issues

1. **Configuration Failures**: Clean the build directory and try again:
   ```bash
   cd build
   make clean
   cmake ..
   ```

2. **Library Not Found**: Ensure both OpenCV and ONNX Runtime are properly installed:
   ```bash
   ls -la opencv/lib/
   ls -la onnxruntime-linux-*/lib/
   ```

## Performance Optimization

The build scripts include several optimizations:

- **OpenCV**: Built with `-O3 -march=native` for maximum performance
- **Architecture Detection**: Automatically optimizes for your specific CPU
- **Parallel Build**: Uses all available CPU cores for faster builds
- **Minimal Build**: Only includes necessary OpenCV modules

## Maintenance

### Updating OpenCV
To update to a newer OpenCV version:
1. Edit `build_opencv.sh` and change `OPENCV_VERSION`
2. Run `./build_opencv.sh --clean`

### Updating ONNX Runtime
To update to a newer ONNX Runtime version:
1. Edit `install_onnxruntime.sh` and change `ONNX_VERSION`
2. Run `./install_onnxruntime.sh --force`

### Cleaning Up
To remove all local installations:
```bash
rm -rf opencv/
rm -rf onnxruntime-linux-*/
rm -rf build/
```

## Notes

- The build system is designed to be self-contained and portable
- All libraries are built locally to avoid system dependency conflicts
- The scripts include comprehensive error checking and user feedback
- Build times can vary significantly depending on your system specifications
- The resulting application is optimized for your specific architecture
