#!/bin/bash

# Standalone Webcam Application Runner
# This script runs the webcam application in standalone mode without IPC dependencies

# Detect architecture and set appropriate library paths
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    # ARM64 architecture
    export LD_LIBRARY_PATH="/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:/home/shkwon/Projects/LVGL/CameraApp/Source/opencv/lib:/home/shkwon/Projects/LVGL/CameraApp/Source/onnxruntime-linux-aarch64-1.16.3/lib:$LD_LIBRARY_PATH"
else
    # x86_64 architecture
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/home/shkwon/Projects/LVGL/CameraApp/Source/opencv/lib:/home/shkwon/Projects/LVGL/CameraApp/Source/onnxruntime-linux-x64-1.16.3/lib:$LD_LIBRARY_PATH"
fi

# Remove any existing socket file to avoid conflicts
rm -f /tmp/webcam_gui_socket

# Navigate to the build directory
cd /home/shkwon/Projects/LVGL/CameraApp/Source/build

# Run the application with simulation mode to avoid camera issues
echo "Starting Webcam Application in Standalone Mode..."
echo "Note: This will run in simulation mode to avoid camera and IPC dependencies"
echo "Press Ctrl+C to stop"
echo ""

# Run with simulation mode to avoid camera and IPC issues
./webcam_app --simulation "$@"
