set(SOURCES
    cudaUtil.cpp
)

# 创建库
add_library(cudaUtil STATIC ${SOURCES})

# 设置包含目录
target_include_directories(cudaUtil PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# 链接CUDA库
target_link_libraries(cudaUtil
    ${CUDA_LIBRARIES}
)

# 安装库
install(TARGETS cudaUtil DESTINATION ${LIBDIR})

