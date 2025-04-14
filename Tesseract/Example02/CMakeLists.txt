cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE Release)

project(opencv-tesseract)

set(CMAKE_CXX_FLAGS "-Wall -O2 -std=c++0x ")

add_executable(opencv-tesseract opencv-tesseract-ocr-example02.cpp)

# find opencv package
find_package(OpenCV REQUIRED)

# includes
set(TESSERACT_INCLUDE_PATH "/usr/include/tesseract")
include_directories(${OpenCV_INCLUDE_DIRS} ${TESSERACT_INCLUDE_PATH})

# libraries
set(TESSERAC_LIBS "tesseract")
set(LIBS ${LIBS} ${OpenCV_LIBS} ${TESSERAC_LIBS})
target_link_libraries(opencv-tesseract ${LIBS})