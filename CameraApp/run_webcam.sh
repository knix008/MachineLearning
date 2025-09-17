#!/bin/bash

# Simple script to run the webcam application directly
# This sets the correct library path to avoid Anaconda conflicts

cd /home/shkwon/Projects/LVGL/CameraApp/Source/build
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./webcam_app
