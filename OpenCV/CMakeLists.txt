cmake_minimum_required(VERSION 3.10)
project(OpenCVTest)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test test01.cpp)
target_link_libraries(test ${OpenCV_LIBS})