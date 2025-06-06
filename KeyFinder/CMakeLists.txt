set(SOURCES
    ConfigFile.cpp
    DeviceManager.cpp
    main.cpp
)

# 创建可执行文件
add_executable(cuBitCrack ${SOURCES})

# 设置包含目录
target_include_directories(cuBitCrack PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/KeyFinderLib
    ${CMAKE_SOURCE_DIR}/CudaKeySearchDevice
    ${CMAKE_SOURCE_DIR}/Logger
)

# 链接库
target_link_libraries(cuBitCrack
    KeyFinderLib
    CudaKeySearchDevice
    cudaUtil
    Logger
    ${CUDA_LIBRARIES}
)

# 安装可执行文件
install(TARGETS cuBitCrack DESTINATION ${BINDIR})

# 如果启用了OpenCL，创建OpenCL版本
if(BUILD_OPENCL)
    add_executable(clBitCrack ${SOURCES})
    
    target_include_directories(clBitCrack PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/KeyFinderLib
        ${CMAKE_SOURCE_DIR}/CLKeySearchDevice
        ${CMAKE_SOURCE_DIR}/Logger
    )
    
    target_link_libraries(clBitCrack
        KeyFinderLib
        CLKeySearchDevice
        clUtil
        Logger
    )
    
    install(TARGETS clBitCrack DESTINATION ${BINDIR})
endif()

