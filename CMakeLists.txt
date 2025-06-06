cmake_minimum_required(VERSION 3.18)
project(BitCrack LANGUAGES C CXX CUDA)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "30;50;60;70;75;80;86;89;90")

# 设置目录
set(LIBDIR ${CMAKE_BINARY_DIR}/lib)
set(BINDIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${LIBDIR})
file(MAKE_DIRECTORY ${BINDIR})

# 设置包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/util
    ${CMAKE_CURRENT_SOURCE_DIR}/AddressUtil
    ${CMAKE_CURRENT_SOURCE_DIR}/CmdParse
    ${CMAKE_CURRENT_SOURCE_DIR}/CryptoUtil
    ${CMAKE_CURRENT_SOURCE_DIR}/KeyFinderLib
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaKeySearchDevice
    ${CMAKE_CURRENT_SOURCE_DIR}/cudaMath
    ${CMAKE_CURRENT_SOURCE_DIR}/cudaUtil
    ${CMAKE_CURRENT_SOURCE_DIR}/secp256k1lib
    ${CMAKE_CURRENT_SOURCE_DIR}/Logger
)

# 添加子目录
add_subdirectory(util)
add_subdirectory(AddressUtil)
add_subdirectory(CmdParse)
add_subdirectory(CryptoUtil)
add_subdirectory(KeyFinderLib)
add_subdirectory(cudaUtil)
add_subdirectory(secp256k1lib)
add_subdirectory(Logger)
add_subdirectory(CudaKeySearchDevice)
add_subdirectory(KeyFinder)
add_subdirectory(cudaInfo)

# 设置构建选项
option(BUILD_OPENCL "Build OpenCL version" OFF)
if(BUILD_OPENCL)
    add_subdirectory(clUtil)
    add_subdirectory(embedcl)
    add_subdirectory(CLKeySearchDevice)
    add_subdirectory(clKeyFinder)
    add_subdirectory(CLUnitTests)
endif()

# 添加AddrGen工具
add_subdirectory(AddrGen)

# 显示CUDA信息
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

