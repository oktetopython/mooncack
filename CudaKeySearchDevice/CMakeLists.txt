set(SOURCES
    CudaAtomicList.cu
    CudaDeviceKeys.cu
    CudaHashLookup.cu
    CudaKeySearchDevice.cpp
    CudaKeySearchDevice.cu
    cudabridge.cu
)

# 创建CUDA库
cuda_add_library(CudaKeySearchDevice STATIC ${SOURCES})

# 设置包含目录
target_include_directories(CudaKeySearchDevice PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cudaMath
)

# 设置链接库
target_link_libraries(CudaKeySearchDevice
    KeyFinderLib
    cudaUtil
    Logger
)

# 安装库
install(TARGETS CudaKeySearchDevice DESTINATION ${LIBDIR})

