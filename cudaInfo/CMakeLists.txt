set(SOURCES
    main.cpp
)

# 创建可执行文件
add_executable(cudaInfo ${SOURCES})

# 设置包含目录
target_include_directories(cudaInfo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cudaUtil
)

# 链接库
target_link_libraries(cudaInfo
    cudaUtil
    ${CUDA_LIBRARIES}
)

# 安装可执行文件
install(TARGETS cudaInfo DESTINATION ${BINDIR})

