cmake_minimum_required(VERSION 3.10)
project(webcam_app CXX)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Architecture detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCHITECTURE "x64")
    set(SYSTEM_LIB_DIR "/lib/x86_64-linux-gnu")
    set(USR_LIB_DIR "/usr/lib/x86_64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCHITECTURE "aarch64")
    set(SYSTEM_LIB_DIR "/lib/aarch64-linux-gnu")
    set(USR_LIB_DIR "/usr/lib/aarch64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l|armv8l")
    set(ARCHITECTURE "aarch64")
    set(SYSTEM_LIB_DIR "/lib/aarch64-linux-gnu")
    set(USR_LIB_DIR "/usr/lib/aarch64-linux-gnu")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. Supported: x86_64, aarch64, arm64")
endif()

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static linking configuration
option(BUILD_STATIC "Build with static linking" OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Static linking flags
if(BUILD_STATIC)
    # Hybrid static linking - use static libraries where available
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".so")
    message(STATUS "Building with hybrid static linking enabled")
endif()

# =============================================================================
# DEPENDENCIES
# =============================================================================

# OpenCV configuration
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/cmake/opencv4")
set(OpenCV_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv4")
set(OpenCV_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib")

# Verify OpenCV installation
if(NOT EXISTS "${OpenCV_DIR}")
    message(FATAL_ERROR "Local OpenCV not found at ${OpenCV_DIR}\nPlease ensure OpenCV is built in the opencv directory.")
endif()

if(NOT EXISTS "${OpenCV_LIBS_DIR}/libopencv_core.so")
    message(FATAL_ERROR "Local OpenCV shared libraries not found at ${OpenCV_LIBS_DIR}\nPlease ensure OpenCV is properly built in the opencv directory.")
endif()

set(OpenCV_FOUND TRUE)
set(OpenCV_VERSION "4.8.1")

# ONNX Runtime configuration
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-${ARCHITECTURE}-1.16.3")

# Verify ONNX Runtime installation
if(NOT EXISTS "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    message(FATAL_ERROR "ONNX Runtime prebuilt library not found at ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so\nPlease run: ./install_onnxruntime.sh to download the correct version for ${ARCHITECTURE}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_c_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found at ${ONNXRUNTIME_DIR}/include/\nPlease run: ./install_onnxruntime.sh to download the correct version for ${ARCHITECTURE}")
endif()

# Model files check
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/yolov8n-face.onnx")
    message(WARNING "YOLOv8 face detection model not found. Please ensure models/yolov8n-face.onnx exists.")
else()
    message(STATUS "✓ YOLOv8 face detection model found")
endif()

# =============================================================================
# TARGET CONFIGURATION
# =============================================================================

# Main executable
add_executable(webcam_app src/main.cpp src/webcam.cpp)

# Include directories
target_include_directories(webcam_app PRIVATE 
    include 
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

# Link libraries
if(BUILD_STATIC)
    # Hybrid static linking - use static libraries where available, shared where not
    target_link_libraries(webcam_app 
        # ONNX Runtime (shared - static not available)
        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
        
        # OpenCV libraries (shared - static not available)
        ${OpenCV_LIBS_DIR}/libopencv_core.so
        ${OpenCV_LIBS_DIR}/libopencv_imgproc.so
        ${OpenCV_LIBS_DIR}/libopencv_highgui.so
        ${OpenCV_LIBS_DIR}/libopencv_imgcodecs.so
        ${OpenCV_LIBS_DIR}/libopencv_videoio.so
        ${OpenCV_LIBS_DIR}/libopencv_dnn.so
        ${OpenCV_LIBS_DIR}/libopencv_features2d.so
        ${OpenCV_LIBS_DIR}/libopencv_flann.so
        
        # System GTK+3/GLib dependencies (shared - static not available)
        ${SYSTEM_LIB_DIR}/libgtk-3.so.0
        ${SYSTEM_LIB_DIR}/libgdk-3.so.0
        ${SYSTEM_LIB_DIR}/libgobject-2.0.so.0
        ${SYSTEM_LIB_DIR}/libglib-2.0.so.0
        ${SYSTEM_LIB_DIR}/libgdk_pixbuf-2.0.so.0
        ${SYSTEM_LIB_DIR}/libcairo.so.2
        ${SYSTEM_LIB_DIR}/libpango-1.0.so.0
        ${SYSTEM_LIB_DIR}/libatk-1.0.so.0
        ${SYSTEM_LIB_DIR}/libpangocairo-1.0.so.0
        ${SYSTEM_LIB_DIR}/libgio-2.0.so.0
        ${SYSTEM_LIB_DIR}/libpangoft2-1.0.so.0
        ${SYSTEM_LIB_DIR}/libharfbuzz.so.0
        ${SYSTEM_LIB_DIR}/libfontconfig.so.1
        ${SYSTEM_LIB_DIR}/libfreetype.so.6
        
        # System libraries (static where possible)
        pthread dl rt m stdc++
    )
    
    # Set RPATH for hybrid static build
    set_target_properties(webcam_app PROPERTIES
        INSTALL_RPATH "${OpenCV_LIBS_DIR}:${ONNXRUNTIME_DIR}/lib:${SYSTEM_LIB_DIR}:${USR_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
        SKIP_BUILD_RPATH FALSE
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
else()
    # Dynamic linking - use shared libraries
    target_link_libraries(webcam_app 
        # ONNX Runtime
        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
        
        # OpenCV libraries
        ${OpenCV_LIBS_DIR}/libopencv_core.so
        ${OpenCV_LIBS_DIR}/libopencv_imgproc.so
        ${OpenCV_LIBS_DIR}/libopencv_highgui.so
        ${OpenCV_LIBS_DIR}/libopencv_imgcodecs.so
        ${OpenCV_LIBS_DIR}/libopencv_videoio.so
        ${OpenCV_LIBS_DIR}/libopencv_dnn.so
        ${OpenCV_LIBS_DIR}/libopencv_features2d.so
        ${OpenCV_LIBS_DIR}/libopencv_flann.so
        
        # System GTK+3/GLib dependencies (explicit paths to avoid Anaconda conflicts)
        ${SYSTEM_LIB_DIR}/libgtk-3.so.0
        ${SYSTEM_LIB_DIR}/libgdk-3.so.0
        ${SYSTEM_LIB_DIR}/libgobject-2.0.so.0
        ${SYSTEM_LIB_DIR}/libglib-2.0.so.0
        ${SYSTEM_LIB_DIR}/libgdk_pixbuf-2.0.so.0
        ${SYSTEM_LIB_DIR}/libcairo.so.2
        ${SYSTEM_LIB_DIR}/libpango-1.0.so.0
        ${SYSTEM_LIB_DIR}/libatk-1.0.so.0
        ${SYSTEM_LIB_DIR}/libpangocairo-1.0.so.0
        ${SYSTEM_LIB_DIR}/libgio-2.0.so.0
        ${SYSTEM_LIB_DIR}/libpangoft2-1.0.so.0
        ${SYSTEM_LIB_DIR}/libharfbuzz.so.0
        ${SYSTEM_LIB_DIR}/libfontconfig.so.1
        ${SYSTEM_LIB_DIR}/libfreetype.so.6
        
        # System libraries
        pthread dl rt m stdc++
    )
endif()

# RPATH configuration
if(BUILD_STATIC)
    # Static builds don't need RPATH
    set_target_properties(webcam_app PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
else()
    # Dynamic builds need RPATH
    set_target_properties(webcam_app PROPERTIES
        INSTALL_RPATH "${OpenCV_LIBS_DIR}:${ONNXRUNTIME_DIR}/lib:${SYSTEM_LIB_DIR}:${USR_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
        SKIP_BUILD_RPATH FALSE
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# Compile definitions
target_compile_definitions(webcam_app PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    VERSION_MAJOR=1 VERSION_MINOR=0 VERSION_PATCH=0
)

# =============================================================================
# BUILD TASKS
# =============================================================================

# Copy model files to build directory
add_custom_command(
    TARGET webcam_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/models
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_BINARY_DIR}/models
    COMMENT "Copying model files to build directory"
)

# =============================================================================
# CUSTOM TARGETS
# =============================================================================

# Run the application
add_custom_target(run-webcam
    COMMAND ${CMAKE_COMMAND} -E echo "Starting webcam application..."
    COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${OpenCV_LIBS_DIR}:${ONNXRUNTIME_DIR}/lib:$ENV{LD_LIBRARY_PATH} ${CMAKE_BINARY_DIR}/webcam_app
    DEPENDS webcam_app
    COMMENT "Running webcam application"
)

# Test target
add_custom_target(test_webcam
    COMMAND ${CMAKE_COMMAND} -E echo "Testing webcam application..."
    COMMAND ${CMAKE_COMMAND} -E echo "Run: make run-webcam"
    DEPENDS webcam_app
)

# Clean build target
add_custom_target(clean-build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target webcam_app
    COMMENT "Clean build completed"
)

# Information target
add_custom_target(info
    COMMAND ${CMAKE_COMMAND} -E echo "=== Build Information ==="
    COMMAND ${CMAKE_COMMAND} -E echo "Architecture: ${ARCHITECTURE}"
    COMMAND ${CMAKE_COMMAND} -E echo "Build directory: ${CMAKE_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "OpenCV shared: ${OpenCV_LIBS_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "ONNX Runtime: ${ONNXRUNTIME_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E echo "Models: ${CMAKE_CURRENT_SOURCE_DIR}/models"
    COMMAND ${CMAKE_COMMAND} -E echo "Executable: ${CMAKE_BINARY_DIR}/webcam_app"
    COMMAND ${CMAKE_COMMAND} -E echo "To run: make run-webcam"
    COMMAND ${CMAKE_COMMAND} -E echo "========================"
    COMMENT "Displaying build information"
)

# =============================================================================
# INSTALLATION
# =============================================================================

install(TARGETS webcam_app RUNTIME DESTINATION bin)
install(DIRECTORY models/ DESTINATION share/webcam_app/models FILES_MATCHING PATTERN "*.onnx" PATTERN "*.pt")

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

message(STATUS "=== Webcam Application Configuration ===")
message(STATUS "Architecture: ${ARCHITECTURE}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR} (1.16.3)")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=========================================")