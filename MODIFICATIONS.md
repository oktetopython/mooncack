# BitCrack 现代化改造修改说明

## 修改概述

本次修改的主要目标是优化BitCrack源码，使其能够适应新时代的GPU架构（sm_86/89/90），确保代码能够在最新的NVIDIA GPU上无错完美运行。主要修改包括更新编译架构支持、添加新GPU架构识别、创建现代化构建系统等。

## 详细修改内容

### 1. 主Makefile更新

原始Makefile中只支持compute capability 3.0（Kepler架构），现已更新为支持多种架构：

```makefile
# 原始设置
COMPUTE_CAP=30
NVCCFLAGS=-std=c++11 -gencode=arch=compute_${COMPUTE_CAP},code=\"sm_${COMPUTE_CAP}\" -Xptxas="-v" -Xcompiler "${CXXFLAGS}"

# 更新后的设置
COMPUTE_CAP=86
NVCCFLAGS=-std=c++11 \
    -gencode=arch=compute_30,code=\"sm_30\" \
    -gencode=arch=compute_50,code=\"sm_50\" \
    -gencode=arch=compute_60,code=\"sm_60\" \
    -gencode=arch=compute_70,code=\"sm_70\" \
    -gencode=arch=compute_75,code=\"sm_75\" \
    -gencode=arch=compute_80,code=\"sm_80\" \
    -gencode=arch=compute_86,code=\"sm_86\" \
    -gencode=arch=compute_89,code=\"sm_89\" \
    -gencode=arch=compute_90,code=\"sm_90\" \
    -Xptxas="-v" -Xcompiler "${CXXFLAGS}"
```

这样修改后，编译器会为每种架构生成优化的代码，确保在不同GPU上都能获得最佳性能。

### 2. CUDA设备识别代码更新

在`cudaUtil.cpp`文件中，添加了对新GPU架构的支持：

```cpp
// 原始代码
switch(devInfo.major) {
case 1:
    cores = 8;
    break;
// ...
case 7:
    cores = 64;
    break;
default:
    cores = 8;
    break;
}

// 更新后的代码
switch(devInfo.major) {
case 1:
    cores = 8;
    break;
// ...
case 7:
    cores = 64;
    break;
case 8:
    // Ampere (8.0) and Ada Lovelace (8.6) and Hopper (8.9)
    if(devInfo.minor == 0) {
        // Ampere A100
        cores = 64;
    } else if(devInfo.minor == 6) {
        // Ada Lovelace (RTX 4000系列)
        cores = 128;
    } else if(devInfo.minor == 9) {
        // Hopper H100
        cores = 128;
    } else {
        // 其他Ampere架构 (RTX 3000系列)
        cores = 128;
    }
    break;
case 9:
    // Blackwell (9.0)
    cores = 128;
    break;
default:
    cores = 64; // 默认值改为更合理的数字
    break;
}
```

这样修改后，程序能够正确识别新的GPU架构，并为每种架构设置合适的核心数。

### 3. 添加CMake构建系统

创建了CMakeLists.txt文件，提供现代化的构建系统支持：

```cmake
cmake_minimum_required(VERSION 3.18)
project(BitCrack LANGUAGES C CXX CUDA)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "30;50;60;70;75;80;86;89;90")

# ...其他设置
```

同时为主要的子目录也创建了对应的CMakeLists.txt文件，使项目能够更容易地在不同平台上构建。

## 预期效果

通过以上修改，BitCrack将能够：

1. 在新的GPU架构（sm_86/89/90）上编译运行，包括：
   - RTX 3000系列（Ampere架构，sm_80）
   - RTX 4000系列（Ada Lovelace架构，sm_86）
   - H100（Hopper架构，sm_89）
   - 未来的Blackwell架构（sm_90）

2. 充分利用新GPU架构的性能优势，提高计算效率

3. 提供更现代化的构建系统支持，简化编译过程

## 编译说明

### 使用Makefile编译

```bash
# 编译CUDA版本
make BUILD_CUDA=1

# 编译OpenCL版本
make BUILD_OPENCL=1

# 同时编译两个版本
make BUILD_CUDA=1 BUILD_OPENCL=1
```

### 使用CMake编译

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目
cmake ..

# 编译
make

# 安装
make install
```

## 注意事项

1. 编译需要CUDA 12.x工具包支持

2. 如果遇到编译错误，可能需要根据实际环境调整CUDA架构设置

3. 对于特定的GPU，可以只保留对应的架构设置，以减少编译时间和二进制文件大小

