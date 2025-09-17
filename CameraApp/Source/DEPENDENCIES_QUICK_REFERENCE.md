# Dependencies Quick Reference

## Essential System Packages (Required)

### Ubuntu/Debian
```bash
sudo apt update && sudo apt install -y \
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
    gstreamer1-plugins-base-devel \
    gtk3-devel \
    glib2-devel \
    cairo-devel \
    pango-devel \
    atk-devel \
    gdk-pixbuf2-devel
```

## What Gets Auto-Installed

✅ **OpenCV 4.8.1** - Built locally with all dependencies  
✅ **ONNX Runtime 1.16.3** - Downloaded for your architecture  
✅ **Image Codecs** - JPEG, PNG, TIFF, WebP (built into OpenCV)  
✅ **Video Codecs** - FFmpeg components (built into OpenCV)  

## What You Need to Install

❌ **Build Tools** - cmake, make, gcc, git, wget, tar, pkg-config  
❌ **GStreamer Dev** - libgstreamer1.0-dev, libgstreamer-plugins-base1.0-dev  
❌ **GTK+3 Dev** - libgtk-3-dev, libglib2.0-dev, libcairo2-dev, libpango1.0-dev, libatk1.0-dev, libgdk-pixbuf2.0-dev  

## Verification Commands

```bash
# Check if all tools are available
cmake --version && make --version && gcc --version

# Check GStreamer development packages
pkg-config --exists gstreamer-1.0 && echo "✓ GStreamer found"

# Check GTK+3 development packages
pkg-config --exists gtk+-3.0 && echo "✓ GTK+3 found"

# Check if build works
cd Source
./build_webcam.sh build
```

## Common Issues

| Issue | Solution |
|-------|----------|
| `cmake: command not found` | `sudo apt install cmake` |
| `make: command not found` | `sudo apt install make` |
| `gcc: command not found` | `sudo apt install build-essential` |
| `pkg-config: command not found` | `sudo apt install pkg-config` |
| `GStreamer not found` | `sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev` |
| `GTK+3 not found` | `sudo apt install libgtk-3-dev libglib2.0-dev libcairo2-dev libpango1.0-dev libatk1.0-dev libgdk-pixbuf2.0-dev` |

## Architecture Support

- **x86_64**: Fully supported
- **aarch64/ARM64**: Fully supported
- **Other**: May require manual configuration

---

**For complete details, see [SYSTEM_DEPENDENCIES.md](SYSTEM_DEPENDENCIES.md)**
