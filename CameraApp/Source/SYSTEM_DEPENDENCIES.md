# System Library Dependencies

This document provides a comprehensive overview of all system library dependencies required for the CameraApp project.

## Overview

The CameraApp project is designed to be as self-contained as possible, with minimal system dependencies. Most libraries (OpenCV, ONNX Runtime) are built or downloaded locally, while only essential system libraries are required from the operating system.

## Quick Installation

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    cmake \
    make \
    build-essential \
    git \
    wget \
    tar \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libatk1.0-dev \
    libgdk-pixbuf2.0-dev
```

### CentOS/RHEL/Fedora
```bash
sudo yum install -y \
    cmake \
    make \
    gcc-c++ \
    git \
    wget \
    tar \
    pkgconfig \
    gstreamer1-devel \
    gstreamer1-plugins-base-devel
```

### Arch Linux
```bash
sudo pacman -S \
    cmake \
    make \
    gcc \
    git \
    wget \
    tar \
    pkg-config \
    gstreamer \
    gst-plugins-base
```

## Detailed Dependencies

### 1. Build Tools (Required)

These tools are essential for building the application:

| Package | Purpose | Installation |
|---------|---------|--------------|
| `cmake` | CMake build system | `apt install cmake` |
| `make` | Make build tool | `apt install make` |
| `gcc/g++` | C/C++ compiler | `apt install build-essential` |
| `git` | Version control (for OpenCV source) | `apt install git` |
| `wget` | Download tool (for ONNX Runtime) | `apt install wget` |
| `tar` | Archive extraction | `apt install tar` |
| `pkg-config` | Library configuration tool | `apt install pkg-config` |

### 2. System Development Libraries (Required)

These are the only system development packages required:

| Package | Purpose | Why Required |
|---------|---------|--------------|
| `libgstreamer1.0-dev` | GStreamer development headers | OpenCV video I/O support |
| `libgstreamer-plugins-base1.0-dev` | GStreamer base plugins dev | GStreamer core functionality |

### 3. Runtime System Libraries

These libraries are automatically available on most Linux distributions and are linked at runtime:

#### Core System Libraries
- `libc.so.6` - C standard library
- `libstdc++.so.6` - C++ standard library
- `libpthread.so.0` - POSIX threads
- `libdl.so.2` - Dynamic linking
- `librt.so.1` - Real-time extensions
- `libm.so.6` - Math library

#### GUI and Display Libraries
- `libgtk-3.so.0` - GTK+ GUI toolkit
- `libgdk-3.so.0` - GDK display library
- `libcairo.so.2` - 2D graphics library
- `libpango-1.0.so.0` - Text layout library
- `libatk-1.0.so.0` - Accessibility toolkit
- `libgdk_pixbuf-2.0.so.0` - Image loading library

#### GStreamer Runtime Libraries
- `libgstreamer-1.0.so.0` - GStreamer core
- `libgstbase-1.0.so.0` - GStreamer base
- `libgstvideo-1.0.so.0` - GStreamer video
- `libgstaudio-1.0.so.0` - GStreamer audio
- `libgstapp-1.0.so.0` - GStreamer application
- `libgstpbutils-1.0.so.0` - GStreamer utilities

#### GLib Libraries
- `libglib-2.0.so.0` - GLib core library
- `libgobject-2.0.so.0` - GObject system
- `libgio-2.0.so.0` - GIO library
- `libgmodule-2.0.so.0` - GModule system

#### X11 Libraries
- `libX11.so.6` - X11 client library
- `libXext.so.6` - X11 extensions
- `libXrender.so.1` - X11 rendering
- `libxcb.so.1` - X11 protocol binding

#### Video/Audio Codec Libraries
- `libavcodec.so.58` - FFmpeg codec library
- `libavformat.so.58` - FFmpeg format library
- `libavutil.so.56` - FFmpeg utilities
- `libswscale.so.5` - FFmpeg scaling library
- `libswresample.so.3` - FFmpeg resampling library

#### Image Format Libraries
- `libjpeg.so.8` - JPEG image format
- `libpng16.so.16` - PNG image format
- `libwebp.so.7` - WebP image format
- `libtiff.so.5` - TIFF image format

#### Compression Libraries
- `libz.so.1` - Zlib compression
- `libbz2.so.1.0` - Bzip2 compression
- `liblzma.so.5` - LZMA compression
- `libzstd.so.1` - Zstandard compression

#### Security and Crypto Libraries
- `libssl.so.3` - OpenSSL SSL/TLS
- `libcrypto.so.3` - OpenSSL crypto
- `libgnutls.so.30` - GnuTLS
- `libgcrypt.so.20` - Libgcrypt

### 4. Optional Python Dependencies

These are only required if you want to convert YOLO models:

| Package | Purpose | Installation |
|---------|---------|--------------|
| `python3` | Python interpreter | `apt install python3` |
| `pip3` | Python package manager | `apt install python3-pip` |
| `torch` | PyTorch framework | `pip3 install torch torchvision` |
| `onnx` | ONNX format support | `pip3 install onnx` |
| `ultralytics` | YOLO model support | `pip3 install ultralytics` |

## Self-Contained Libraries

The following libraries are built or downloaded locally and do not require system installation:

### OpenCV 4.8.1 (Local Build)
- **Built by**: `build_opencv.sh`
- **Location**: `Source/opencv/`
- **Includes**: All OpenCV modules, image codecs, video codecs
- **Self-contained**: JPEG, PNG, TIFF, WebP, Zlib, FFmpeg components

### ONNX Runtime 1.16.3 (Local Download)
- **Downloaded by**: `install_onnxruntime.sh`
- **Location**: `Source/onnxruntime-linux-{arch}-1.16.3/`
- **Architecture**: Auto-detected (x64/aarch64)
- **Self-contained**: Complete ONNX Runtime library

## Dependency Categories

| Category | Required | Auto-Installed | System Only | Size |
|----------|----------|----------------|-------------|------|
| **Build Tools** | ✅ | ❌ | ✅ | ~50MB |
| **GStreamer Dev** | ✅ | ❌ | ✅ | ~20MB |
| **OpenCV** | ✅ | ✅ | ❌ | ~71MB |
| **ONNX Runtime** | ✅ | ✅ | ❌ | ~50MB |
| **System Libraries** | ✅ | ❌ | ✅ | ~200MB |
| **Python Tools** | ❌ | ❌ | ✅ | ~500MB |

## Architecture Support

The build system supports multiple architectures:

### x86_64 (x64)
- **ONNX Runtime**: `onnxruntime-linux-x64-1.16.3`
- **OpenCV**: Optimized for x64 with `-march=native`
- **System Libraries**: Standard x86_64 libraries

### aarch64 (ARM64)
- **ONNX Runtime**: `onnxruntime-linux-aarch64-1.16.3`
- **OpenCV**: Optimized for aarch64 with `-march=native`
- **System Libraries**: Standard aarch64 libraries

## Verification Commands

### Check Build Tools
```bash
# Verify essential tools
cmake --version
make --version
gcc --version
git --version
wget --version
pkg-config --version
```

### Check System Dependencies
```bash
# Check GStreamer development packages
pkg-config --exists gstreamer-1.0 && echo "✓ GStreamer 1.0 found"
pkg-config --exists gstreamer-base-1.0 && echo "✓ GStreamer base found"
```

### Check Runtime Libraries
```bash
# Check if executable can find all libraries
cd Source/build
ldd webcam_app | grep -E "(not found|=>)"
```

### Check Local Libraries
```bash
# Verify OpenCV installation
ls -la Source/opencv/lib/libopencv_core.so*

# Verify ONNX Runtime installation
ls -la Source/onnxruntime-linux-*/lib/libonnxruntime.so*
```

## Troubleshooting

### Missing Build Tools
```bash
# Install missing build tools
sudo apt update
sudo apt install -y cmake make build-essential git wget tar pkg-config
```

### Missing GStreamer Development
```bash
# Install GStreamer development packages
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### Library Not Found Errors
```bash
# Check library paths
echo $LD_LIBRARY_PATH
ldd Source/build/webcam_app

# Rebuild with proper paths
cd Source
./build_webcam.sh clean
./build_webcam.sh build
```

### Architecture Mismatch
```bash
# Check system architecture
uname -m

# Verify ONNX Runtime architecture
ls -la Source/onnxruntime-linux-*/
```

## Performance Considerations

### Build Time
- **OpenCV Build**: ~30-60 minutes (first time)
- **Application Build**: ~2-5 minutes
- **Dependency Download**: ~5-10 minutes

### Disk Space
- **OpenCV**: ~71MB installed
- **ONNX Runtime**: ~50MB installed
- **Build artifacts**: ~500MB temporary
- **Total**: ~1GB for complete installation

### Memory Usage
- **Build Process**: 2-4GB RAM recommended
- **Runtime**: 512MB-2GB RAM depending on model size

## Security Considerations

### System Libraries
- All system libraries are from official distribution repositories
- Regular security updates through system package manager
- No custom system library modifications

### Local Libraries
- OpenCV: Built from official source with security patches
- ONNX Runtime: Downloaded from official Microsoft releases
- All libraries verified with checksums

## Maintenance

### Updating System Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Rebuild local libraries if needed
cd Source
./build_opencv.sh --clean
./install_onnxruntime.sh --force
```

### Cleaning Build Artifacts
```bash
# Clean build directories
cd Source
./build_webcam.sh clean
rm -rf opencv_build/ opencv_src/
```

## References

- [OpenCV Installation Guide](https://docs.opencv.org/4.8.1/d7/d9f/tutorial_linux_install.html)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [GStreamer Development](https://gstreamer.freedesktop.org/documentation/application-development/basics/index.html)
- [CMake Documentation](https://cmake.org/documentation/)

---

**Note**: This document reflects the current state of the CameraApp project. Dependencies may change with future updates.
