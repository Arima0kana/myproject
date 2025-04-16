cmake_minimum_required(VERSION 3.10)
project(AprilTagDetection)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

# 查找 AprilTag 库
find_package(apriltag REQUIRED)

# 添加可执行文件
add_executable(AprilTag_2_1 AprilTag_2_1.cpp)

# 添加头文件路径
target_include_directories(AprilTag_2_1 PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${apriltag_INCLUDE_DIRS}
)

# 链接所需库
target_link_libraries(AprilTag_2_1 PRIVATE
    ${OpenCV_LIBS}
    apriltag::apriltag
    pthread  # 串口、线程支持
)

# Raspberry Pi 编译优化
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    target_compile_options(AprilTag_2_1 PRIVATE
        -O2
        -march=armv8-a+crc
        -mtune=cortex-a72
        -mfpu=neon-fp-armv8
        -mfloat-abi=hard
    )
    message(STATUS "Enabling Raspberry Pi optimizations")
endif()
