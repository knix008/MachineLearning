cmake_minimum_required(VERSION 3.10)
project(OpenCVUpscaling)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 기존 테스트 실행 파일
add_executable(test test01.cpp)
target_link_libraries(test ${OpenCV_LIBS})

# 간단한 4배 확대 실행 파일
add_executable(simple_upscale_4x simple_upscale_4x.cpp)
target_link_libraries(simple_upscale_4x ${OpenCV_LIBS})

# 고급 4배 확대 실행 파일
add_executable(upscale_4x upscale_4x.cpp)
target_link_libraries(upscale_4x ${OpenCV_LIBS})

# 기존 얼굴 검출 실행 파일
add_executable(face_detection OpenCVFaceDetectionExample01.cpp)
target_link_libraries(face_detection ${OpenCV_LIBS})