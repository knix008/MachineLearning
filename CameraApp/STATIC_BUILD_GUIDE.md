# Static Build Guide for CameraApp

This guide explains how to build the CameraApp with hybrid static linking to create a more portable executable with reduced external dependencies.

## 🎯 **What is Hybrid Static Linking?**

Hybrid static linking combines static and dynamic linking approaches:
- **Static linking** for system libraries where possible
- **Dynamic linking** for complex libraries (OpenCV, ONNX Runtime) that don't have static versions available
- **RPATH embedding** to ensure the executable finds its dependencies

### **Benefits:**
- ✅ **Reduced dependencies** - Fewer external libraries required
- ✅ **Better portability** - Can run on systems with minimal library installations
- ✅ **Embedded paths** - RPATH ensures correct library loading
- ✅ **Smaller than full static** - More reasonable file size

### **Drawbacks:**
- ❌ **Not fully standalone** - Still requires some shared libraries
- ❌ **Library path dependency** - Needs correct LD_LIBRARY_PATH or RPATH
- ❌ **Complex setup** - Requires careful library path management

## 📋 **Prerequisites**

### **1. Install Static Development Libraries**

```bash
# Install system static libraries
sudo apt update
sudo apt install -y \
    libgtk-3-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libatk1.0-dev \
    libgdk-pixbuf2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxcb-dev
```

### **2. Build OpenCV with Static Libraries**

The current OpenCV build creates shared libraries. For static linking, you need to rebuild OpenCV:

```bash
cd Source
# Modify build_opencv.sh to use static libraries
# Add these CMake flags to the OpenCV build:
# -DBUILD_SHARED_LIBS=OFF
# -DBUILD_STATIC_LIBS=ON
./build_opencv.sh --clean
```

### **3. Get ONNX Runtime Static Libraries**

ONNX Runtime provides static libraries. Download the static version:

```bash
cd Source
# Download ONNX Runtime static libraries
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
```

## 🚀 **Building with Static Linking**

### **Method 1: Using the Static Build Script (Recommended)**

```bash
cd Source
./build_static.sh
```

### **Method 2: Manual CMake Configuration**

```bash
cd Source
mkdir build-static && cd build-static
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON ..
make -j$(nproc)
```

### **Method 3: Using the Build Script with Static Flag**

```bash
cd Source
./build_webcam.sh build-static
```

## 🔧 **CMake Configuration for Static Build**

The CMakeLists.txt includes a `BUILD_STATIC` option:

```cmake
# Enable static linking
option(BUILD_STATIC "Build with static linking" OFF)

# Static linking flags
if(BUILD_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
endif()
```

## 📊 **Expected Results**

### **Dynamic Build:**
- **Size**: ~2-5MB executable
- **Dependencies**: Requires system libraries
- **Distribution**: Need to include libraries

### **Hybrid Static Build:**
- **Size**: ~100KB executable (much smaller than full static)
- **Dependencies**: OpenCV, ONNX Runtime, system GTK+3 libraries
- **Distribution**: Executable + library files
- **Portability**: Better than dynamic, not as good as full static

## 🛠️ **Troubleshooting**

### **Missing Static Libraries**
```bash
# Check for static libraries
ls /usr/lib/x86_64-linux-gnu/*.a | grep -E "(gtk|glib|cairo)"

# Install missing packages
sudo apt install -y <missing-package>-dev
```

### **OpenCV Static Build Issues**
```bash
# Clean and rebuild OpenCV
cd Source
rm -rf opencv opencv_build opencv_src
./build_opencv.sh --clean
```

### **ONNX Runtime Static Issues**
```bash
# Download correct static version
cd Source
rm -rf onnxruntime-linux-*
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
```

### **Linker Errors**
```bash
# Check for missing static libraries
ldd build/webcam_app

# Install additional static libraries
sudo apt install -y libx11-dev libxext-dev libxrender-dev
```

## 📁 **File Structure After Static Build**

```
Source/
├── build/
│   └── webcam_app          # Standalone executable (~50-100MB)
├── models/
│   └── yolov8n-face.onnx    # AI model (still needed)
└── ...
```

## 🎯 **Running the Hybrid Static Executable**

### **Method 1: Using the Static Runner Script (Recommended)**
```bash
cd Source
./run_static.sh
```

### **Method 2: Manual Library Path Setup**
```bash
cd Source/build
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/home/shkwon/Projects/LVGL/CameraApp/Source/opencv/lib:/home/shkwon/Projects/LVGL/CameraApp/Source/onnxruntime-linux-x64-1.16.3/lib:$LD_LIBRARY_PATH"
./webcam_app
```

### **Method 3: Using the Build Script**
```bash
cd Source
./build_webcam.sh run  # This will use the correct library paths
```

## 📈 **Performance Comparison**

| Aspect | Dynamic | Static |
|--------|---------|--------|
| **Startup Time** | Fast | Slower |
| **Memory Usage** | Lower | Higher |
| **File Size** | Small | Large |
| **Distribution** | Complex | Simple |
| **Dependencies** | Many | None |

## 🔍 **Verifying Static Build**

```bash
# Check if truly static
ldd build/webcam_app
# Should show minimal system libraries only

# Check file size
ls -lh build/webcam_app

# Test on clean system
# Copy to system without development libraries
# Should still run
```

## 📝 **Notes**

- **OpenCV**: May need custom build for static libraries
- **ONNX Runtime**: Download static version from GitHub releases
- **GTK+3**: Static libraries may not be available on all distributions
- **GStreamer**: Static linking can be complex due to plugin system

## 🚨 **Important Considerations**

1. **License Compliance**: Ensure all static libraries are compatible with your license
2. **Size Limits**: Some systems have executable size limits
3. **Debugging**: Static builds are harder to debug
4. **Updates**: Need to rebuild entire executable for library updates

---

**For most use cases, dynamic linking is recommended unless you specifically need a standalone executable.**
