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
# This list covers a wide range of common and newer GPUs, including:
# Kepler (30), Maxwell (50), Pascal (60), Volta (70), Turing (75),
# Ampere (80, 86), Lovelace (89), Hopper (90).
# Update this list as new GPU architectures become common.
set(CMAKE_CUDA_ARCHITECTURES "30;50;60;70;75;80;86;89;90")

# Option to build a shared library
option(BUILD_SHARED_LIB "Build a shared library (bitcrack_shared)" OFF)

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

# Shared library target
if(BUILD_SHARED_LIB)
    message(STATUS "Building shared library bitcrack_shared")

    # Define source files for the shared library
    set(BITCRACK_SHARED_SRCS
        # KeyFinderLib
        KeyFinderLib/KeyFinder.cpp
        # CudaKeySearchDevice
        CudaKeySearchDevice/CudaAtomicList.cu
        CudaKeySearchDevice/CudaDeviceKeys.cu
        CudaKeySearchDevice/CudaHashLookup.cu
        CudaKeySearchDevice/CudaKeySearchDevice.cpp
        CudaKeySearchDevice/CudaKeySearchDevice.cu
        CudaKeySearchDevice/cudabridge.cu
        # CryptoUtil
        CryptoUtil/Rng.cpp
        CryptoUtil/checksum.cpp
        CryptoUtil/hash.cpp # CryptoUtil version
        CryptoUtil/ripemd160.cpp
        CryptoUtil/sha256.cpp
        # AddressUtil
        AddressUtil/Base58.cpp
        AddressUtil/hash.cpp # AddressUtil version
        # secp256k1lib
        secp256k1lib/secp256k1.cpp
        # cudaUtil
        cudaUtil/cudaUtil.cpp
    )

    # cudaMath is header-only, its path is already in include_directories

    add_library(bitcrack_shared SHARED ${BITCRACK_SHARED_SRCS})

    # Set C++ standard for the shared library
    set_target_properties(bitcrack_shared PROPERTIES
        CXX_STANDARD ${CMAKE_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED}
        CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
        CUDA_STANDARD_REQUIRED ${CMAKE_CUDA_STANDARD_REQUIRED}
        POSITION_INDEPENDENT_CODE ON # Required for shared libraries
    )

    # Set visibility presets
    set(CMAKE_CUDA_VISIBILITY_PRESET "hidden")
    set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

    # Generate export header for symbol visibility
    include(GenerateExportHeader)
    generate_export_header(bitcrack_shared
        BASE_NAME BITCRACK
        EXPORT_MACRO_NAME BITCRACK_API
        EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/bitcrack_export.h
    )

    target_include_directories(bitcrack_shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CURRENT_BINARY_DIR}/include # For the generated export header
    )

    target_link_libraries(bitcrack_shared PRIVATE CUDA::cudart)

    # Ensure the main executable can link against CUDA runtime if needed,
    # even if it doesn't directly use CUDA source files itself (though KeyFinder does)
    # This is generally good practice.
    # The individual subdirectories like CudaKeySearchDevice already link to CUDA::cudart
    # so this might be redundant here but ensures linkage for bitcrack_shared itself.

endif()

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
# Ensure AddrGen also has access to necessary headers if it depends on any of the shared lib components
# For now, assuming AddrGen is independent or handles its dependencies.
add_subdirectory(AddrGen)

# 显示CUDA信息
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

