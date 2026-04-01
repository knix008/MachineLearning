cmake_minimum_required(VERSION 3.16)
project(image_upscaler_gtk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK4 REQUIRED IMPORTED_TARGET gtk4)
find_package(CURL REQUIRED)

# ONNX Runtime: set -DONNXRUNTIME_ROOT=/path or CMAKE_PREFIX_PATH
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
if(ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
  find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
else()
  find_path(
    ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
      /opt/homebrew/opt/onnxruntime/include
      /usr/local/opt/onnxruntime/include
      /usr/include/onnxruntime
      "${CMAKE_PREFIX_PATH}/include"
  )
  find_library(
    ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS
      /opt/homebrew/opt/onnxruntime/lib
      /usr/local/opt/onnxruntime/lib
      /usr/lib/x86_64-linux-gnu
      "${CMAKE_PREFIX_PATH}/lib"
  )
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(
    FATAL_ERROR
      "ONNX Runtime not found. Install it (e.g. brew install onnxruntime on macOS) "
      "or set ONNXRUNTIME_ROOT to the extracted release folder (include/ + lib/)."
  )
endif()

add_executable(
  image_upscaler_gtk
  src/main.cpp
  src/onnx_sr.cpp
)

target_include_directories(image_upscaler_gtk PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(
  image_upscaler_gtk
  PRIVATE
    PkgConfig::GTK4
    CURL::libcurl
    "${ONNXRUNTIME_LIB}"
)

if(APPLE)
  target_link_libraries(image_upscaler_gtk PRIVATE "-framework Foundation")
endif()
