#!/bin/bash

# Static Build Runner Script
# This script runs the hybrid static build with proper library paths

# Detect architecture and set appropriate library paths
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    # ARM64 architecture
    export LD_LIBRARY_PATH="/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:/home/shkwon/Projects/LVGL/CameraApp/Source/opencv/lib:/home/shkwon/Projects/LVGL/CameraApp/Source/onnxruntime-linux-aarch64-1.16.3/lib:$LD_LIBRARY_PATH"
else
    # x86_64 architecture
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/home/shkwon/Projects/LVGL/CameraApp/Source/opencv/lib:/home/shkwon/Projects/LVGL/CameraApp/Source/onnxruntime-linux-x64-1.16.3/lib:$LD_LIBRARY_PATH"
fi

# Navigate to the build directory and execute the application
cd /home/shkwon/Projects/LVGL/CameraApp/Source/build
./webcam_app "$@"
