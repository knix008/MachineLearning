# CameraApp - Real-time Face Detection

A real-time face detection application using OpenCV and YOLOv8 with ONNX Runtime 1.16.3.

## Features

- Real-time face detection using webcam
- YOLOv8n-face model for accurate face detection
- ONNX Runtime 1.16.3 for optimized inference
- Cross-platform support (Linux, Windows, macOS)
- **Multi-architecture support** (x86_64, ARM64/Apple Silicon)
- Self-contained build system using CMake
- **Automated dependency installation** - No manual dependency installation required**
- Self-contained libraries (OpenCV, ONNX Runtime)
- **Static linking support** for portable executables

## Prerequisites

- Linux system (Ubuntu/Debian, CentOS/RHEL/Fedora, Arch Linux)
- **Architecture support**: x86_64, ARM64 (Apple Silicon), aarch64
- Webcam device
- Internet connection (for downloading dependencies)
- sudo access (for installing system packages)

**All other dependencies are automatically installed by the build system!**

### **Apple Silicon Support**
- ✅ **Native ARM64 Linux** (Asahi Linux, Ubuntu ARM64)
- ✅ **Virtualized ARM64** (VMware, Parallels, QEMU)
- ✅ **Cross-compilation** from x86_64 to ARM64

**For detailed system dependencies, see [SYSTEM_DEPENDENCIES.md](Source/SYSTEM_DEPENDENCIES.md)**

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CameraApp
   ```

2. **Build and run the application (auto-installs all dependencies):**
   ```bash
   ./run.sh build
   ./run.sh run
   ```

   **Or in one command:**
   ```bash
   ./run.sh run
   ```

3. **Alternative: Install dependencies first, then build:**
   ```bash
   ./run.sh deps    # Install system dependencies
   ./run.sh build   # Build the application
   ./run.sh run     # Run the application
   ```

## Build System

The project uses a comprehensive build system with automated dependency management:

### 🚀 **Automated Build Scripts**

1. **run.sh** - Root-level convenience script that:
   - Provides easy access to all build commands
   - Handles dependency installation automatically
   - Works from the project root directory

2. **build_webcam.sh** - Main build script that:
   - **Auto-installs system dependencies** (GTK+3, GStreamer, build tools)
   - Auto-installs ONNX Runtime if missing
   - Auto-builds OpenCV locally if missing
   - Builds the webcam application
   - Provides run, test, and clean commands

3. **install_dependencies.sh** - System dependency installer that:
   - Detects Linux distribution (Ubuntu/Debian, CentOS/RHEL/Fedora, Arch)
   - Installs all required system packages automatically
   - Supports selective installation (build tools, GStreamer, GTK+3)
   - Verifies all installations

4. **build_opencv.sh** - OpenCV build script that:
   - Detects CPU architecture (x64/aarch64)
   - Downloads and builds OpenCV 4.8.1 locally
   - Includes self-contained image and video codecs
   - Optimized for the target architecture

5. **install_onnxruntime.sh** - ONNX Runtime installer that:
   - Detects CPU architecture (x64/aarch64)
   - Downloads appropriate ONNX Runtime version
   - Installs to the correct location

### 🎯 **Self-Contained Dependencies**

The build system is designed to be completely self-contained:
- **System Dependencies**: Auto-installed (GTK+3, GStreamer, build tools)
- **OpenCV**: Built locally with all dependencies included
- **ONNX Runtime**: Downloaded and installed locally
- **No manual dependency installation required!**

## Installation

### 🎯 **Automated Installation (Recommended)**

The project includes fully automated scripts that handle **all dependencies**:

```bash
# From project root - installs everything automatically
./run.sh build
```

This single command will:
1. **Install system dependencies** (GTK+3, GStreamer, build tools)
2. Install ONNX Runtime 1.16.3 (if missing)
3. Build OpenCV 4.8.1 locally (if missing)
4. Build the webcam application
5. Set up all necessary configurations

### 🔧 **Step-by-Step Installation**

If you prefer to install dependencies separately:

1. **Install system dependencies:**
   ```bash
   ./run.sh deps
   ```

2. **Build the application:**
   ```bash
   ./run.sh build
   ```

3. **Run the application:**
   ```bash
   ./run.sh run
   ```

### 🛠️ **Manual Installation (Advanced)**

If you prefer manual installation:

1. **Install system dependencies:**
   ```bash
   cd Source
   ./install_dependencies.sh
   ```

2. **Install ONNX Runtime:**
   ```bash
   cd Source
   ./install_onnxruntime.sh
   ```

3. **Build OpenCV:**
   ```bash
   cd Source
   ./build_opencv.sh
   ```

4. **Build the application:**
   ```bash
   cd Source
   mkdir build && cd build
   cmake ..
   make
   ```

## Building the Application

### 🚀 **Method 1: Automated Build (Recommended)**
```bash
# From project root - handles everything automatically
./run.sh build
```

### 🔧 **Method 2: Step-by-Step Build**
```bash
# Install dependencies first
./run.sh deps

# Then build
./run.sh build
```

### 🛠️ **Method 3: Manual Build (Advanced)**
```bash
cd Source
./build_webcam.sh build
```

### 📋 **Method 4: Complete Manual Build**
```bash
cd Source
./install_dependencies.sh
./install_onnxruntime.sh
./build_opencv.sh
mkdir build && cd build
cmake ..
make
```

The automated build process will:
- **Auto-install system dependencies** (GTK+3, GStreamer, build tools)
- Auto-install missing ONNX Runtime
- Auto-build OpenCV locally if missing
- Compile the C++ source code
- Link against OpenCV and ONNX Runtime libraries
- Copy model files to the build directory
- Set up proper RPATH for self-contained execution

## Running the Application

### 🚀 **Method 1: Automated Run (Recommended)**
```bash
# From project root - builds and runs automatically
./run.sh run
```

### 🔧 **Method 2: Build then Run**
```bash
./run.sh build
./run.sh run
```

### 🛠️ **Method 3: Manual Run**
```bash
cd Source
./build_webcam.sh run
```

### 📋 **Method 4: Direct Execution**
```bash
cd Source/build
make run-webcam
# OR
./webcam_app
```

The executable is configured with RPATH, so it should find the libraries automatically.

## Available Commands

### 🚀 **Root Script Commands (Recommended)**
```bash
./run.sh deps      # Install system dependencies only
./run.sh build     # Build the application (auto-installs deps)
./run.sh build-static  # Build with static linking (portable executable)
./run.sh run       # Build and run the application
./run.sh test      # Run tests
./run.sh clean     # Clean build files
./run.sh info      # Show build information
./run.sh help      # Show help
```

### 🏗️ **Architecture-Specific Builds**
```bash
# Automatic architecture detection (x86_64, ARM64/Apple Silicon)
./run.sh build          # Detects architecture automatically
./run.sh build-static   # Static linking for portable executables

# Manual architecture specification (advanced)
cd Source
cmake -DCMAKE_SYSTEM_PROCESSOR=aarch64 ..  # Force ARM64
cmake -DCMAKE_SYSTEM_PROCESSOR=x86_64 ..   # Force x86_64
```

### 🔧 **Source Script Commands**
```bash
cd Source
./build_webcam.sh deps         # Install system dependencies
./build_webcam.sh build        # Build the application
./build_webcam.sh run          # Build and run the application
./build_webcam.sh test         # Run tests
./build_webcam.sh clean        # Clean build files
./build_webcam.sh info         # Show build information
./build_webcam.sh help         # Show help
```

### 🛠️ **Dependency Installation Commands**
```bash
cd Source
./install_dependencies.sh                    # Install all dependencies
./install_dependencies.sh --build-only      # Install build tools only
./install_dependencies.sh --gstreamer       # Install GStreamer only
./install_dependencies.sh --gtk3            # Install GTK+3 only
```

### 📋 **CMake Targets**
- `make` - Build the application
- `make run-webcam` - Build and run the application
- `make info` - Display build configuration information
- `make clean` - Clean build files
- `make clean-build` - Complete clean rebuild
- `make test_webcam` - Run basic tests

## Configuration

The application uses the following configuration:

- **OpenCV**: Local installation in `Source/opencv/`
- **ONNX Runtime**: Prebuilt 1.16.3 in `Source/onnxruntime-linux-x64-1.16.3/`
- **Model**: YOLOv8n-face model in `Source/models/`
- **RPATH**: Automatically configured for self-contained execution

## Troubleshooting

### 🔧 **System Dependency Issues**
```bash
# Install missing system dependencies
./run.sh deps

# Or install manually
cd Source
./install_dependencies.sh
```

### 📦 **ONNX Runtime Installation Issues**
```bash
# Check if the installation script exists
ls -la Source/install_onnxruntime.sh

# Make it executable if needed
chmod +x Source/install_onnxruntime.sh

# Run with verbose output
./install_onnxruntime.sh --force
```

### 🔗 **Library Not Found Errors**
If you encounter library not found errors, ensure:
1. System dependencies are installed: `./run.sh deps`
2. OpenCV is properly built in `Source/opencv/`
3. ONNX Runtime is available in `Source/onnxruntime-linux-x64-1.16.3/`
4. Use `./run.sh run` instead of direct execution

### 📹 **Webcam Access Issues**
- Ensure your webcam is connected and accessible
- Check webcam permissions on your system
- Try running with `sudo` if needed (for device access)

### 🤖 **Model Loading Issues**
- Verify `yolov8n-face.onnx` exists in `Source/models/`
- Check file permissions on the model file

### 🐛 **Build Issues**
```bash
# Clean and rebuild
./run.sh clean
./run.sh build

# Or step by step
./run.sh deps
./run.sh build
```

## Project Structure

```
CameraApp/
├── run.sh                                      # Root convenience script
├── Source/
│   ├── build/                                    # Build directory
│   ├── include/                                  # Header files
│   │   └── webcam.h
│   ├── src/                                     # Source files
│   │   ├── main.cpp
│   │   └── webcam.cpp
│   ├── models/                                  # AI models
│   │   └── yolov8n-face.onnx
│   ├── opencv/                                  # Local OpenCV installation
│   │   ├── lib/                                # OpenCV libraries
│   │   ├── include/                            # OpenCV headers
│   │   └── share/                              # OpenCV data files
│   ├── onnxruntime-linux-x64-1.16.3/           # ONNX Runtime 1.16.3
│   │   ├── lib/
│   │   │   └── libonnxruntime.so
│   │   └── include/
│   │       └── onnxruntime_c_api.h
│   ├── build_webcam.sh                         # Main build script
│   ├── install_dependencies.sh                 # System dependency installer
│   ├── build_opencv.sh                         # OpenCV build script
│   ├── install_onnxruntime.sh                  # ONNX Runtime installer
│   ├── convert_yolo.sh                         # Model conversion script
│   ├── BUILD_SYSTEM.md                         # Build system documentation
│   ├── SYSTEM_DEPENDENCIES.md                  # System dependencies guide
│   ├── DEPENDENCIES_QUICK_REFERENCE.md         # Quick dependency reference
│   ├── LIBRARY_ANALYSIS.md                     # Comprehensive library analysis
│   ├── LIBRARY_SUMMARY.md                      # Library summary table
│   └── CMakeLists.txt                          # Build configuration
├── .gitignore
└── README.md
```

## Development

### Adding New Features
1. Add source files to `Source/src/`
2. Add headers to `Source/include/`
3. Update `CMakeLists.txt` if needed
4. Rebuild with `make`

### Debugging
```bash
cd Source/build
make clean
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

### Updating ONNX Runtime
To update to a newer version of ONNX Runtime:
1. Edit `Source/install_onnxruntime.sh` and change the `ONNX_VERSION` variable
2. Run `./install_onnxruntime.sh --force`
3. Update `Source/CMakeLists.txt` if the directory structure changes

## Dependencies

### 🎯 **Automatically Installed Dependencies**

**System Dependencies (Auto-installed):**
- **Build Tools**: cmake, make, gcc, git, wget, pkg-config
- **GStreamer**: libgstreamer1.0-dev, libgstreamer-plugins-base1.0-dev
- **GTK+3**: libgtk-3-dev, libglib2.0-dev, libcairo2-dev, libpango1.0-dev, libatk1.0-dev, libgdk-pixbuf2.0-dev

**Application Dependencies (Auto-installed):**
- **OpenCV 4.8.1** - Computer vision library (built locally)
- **ONNX Runtime 1.16.3** - Machine learning inference engine (downloaded)
- **YOLOv8n-face** - Face detection model (provided)

**Build System:**
- **CMake 3.10+** - Build system (auto-installed)
- **C++14** - Programming language standard

## License

This project is licensed under the MIT License - see the LICENSE file for details.
