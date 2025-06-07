[English](#english-version) | [中文](#chinese-version)

<a name="english-version"></a>
## English Version
# BitCrack

A tool for brute-forcing Bitcoin private keys. The main purpose of this project is to contribute to the effort of solving the [Bitcoin puzzle transaction](https://blockchain.info/tx/08389f34c98c606322740c0be6a7125d9860bb8d5cb182c02f98461e5fa6cd15): A transaction with 32 addresses that become increasingly difficult to crack.

## Key Features and Optimizations (Post-Modernization)

This version of BitCrack includes numerous enhancements:

*   **Modern Build System**: Uses CMake (version 3.18+) for cross-platform compilation, supporting modern C++17 compilers and simplifying dependency management (e.g., `spdlog`, `pybind11` fetched automatically).
*   **Enhanced GPU Support**:
    *   Support for newer CUDA architectures (e.g., Ampere, Lovelace, Hopper included by default).
    *   Dynamic adjustment of CUDA kernel launch parameters (blocks, threads) based on detected GPU capabilities for optimized out-of-the-box performance.
*   **Efficient ECC Operations**: Implemented fixed-window ECC point multiplication for `k*G` on both CPU (for utility functions) and GPU (for high-performance public key generation), significantly speeding up this core cryptographic step.
*   **Improved Address Handling**:
    *   Robust Base58Check decoding allows correct parsing and HASH160 extraction for various Bitcoin address types, including P2PKH (starting with '1') and P2SH (starting with '3').
    *   CPU-side target verification (after GPU Bloom filter) uses `std::unordered_set` for high efficiency with large address lists (millions of targets).
*   **Advanced Performance Optimizations**:
    *   **CUDA Streams**: Utilized to pipeline GPU kernel execution, device-to-host data transfers (results), and CPU-side result processing, enabling better operational overlap and increased throughput.
    *   **Multi-GPU Support**: Allows distributing the workload across several CUDA devices specified on the command line, significantly increasing the overall key search rate.
*   **Cross-Language Compatibility & Extensibility**:
    *   **Shared Library**: Core BitCrack logic can be built as a shared library (`libbitcrack_shared.so` or `bitcrack_shared.dll`).
    *   **C API**: A C-language API (`bitcrack_capi.h`) exposes core functionalities (session management, configuration, async search, result polling) for integration into other software.
    *   **Python Bindings**: A Python module (`bitcrack_python`) built with `pybind11` allows controlling BitCrack and retrieving results directly from Python scripts.
    *   **Rust Bindings**: A Rust crate (`bitcrack_rs_bindings`) built with `bindgen` provides safe Rust wrappers around the C API for use in Rust applications.
*   **Improved Logging & Recovery**:
    *   Integrated `spdlog` for structured, leveled logging (to console and optionally to file). Log level (trace, debug, info, warn, error, critical, off) and log file path are configurable via CLI.
    *   Robust multi-GPU aware checkpointing saves and restores the progress for each active GPU independently, allowing scans to be resumed. Checkpoint interval is configurable.

## Installation

### Prerequisites

*   **CMake**: Version 3.18 or higher. ([Download CMake](https://cmake.org/download/))
*   **C++ Compiler**: A C++17 compatible compiler (e.g., MSVC on Windows, GCC or Clang on Linux).
*   **CUDA Toolkit**: Version 10.1 or newer is recommended for CUDA support. This includes the NVIDIA GPU drivers and the `nvcc` compiler. ([Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads))
*   **OpenCL SDK (Optional)**: If you intend to build or use the OpenCL version (`clBitCrack`). This can often be found in GPU vendor driver packages (AMD, Intel, NVIDIA) or sometimes within the CUDA Toolkit.
*   **Python (Optional, for Python bindings)**: Python 3.6 or higher, including development headers (e.g., `python3-dev` on Debian/Ubuntu). Pip is also useful for Python package management.
*   **Rust (Optional, for Rust bindings)**: The Rust toolchain (rustc, cargo). ([Install Rust](https://www.rust-lang.org/tools/install))
*   **Git**: For cloning the repository.

### Building with CMake (Recommended Method)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/brichard19/BitCrack.git # Or your fork/source location
    cd BitCrack
    ```

2.  **Configure with CMake:**
    It's best practice to create a separate build directory.
    ```bash
    mkdir build
    cd build
    ```

    Run CMake to configure the project. Here are some common configuration examples:

    *   **Default (CUDA, Release mode):**
        ```bash
        cmake ..
        # For a specific build type like Release or Debug:
        # cmake -DCMAKE_BUILD_TYPE=Release ..
        ```
        *You can specify CUDA architectures to build for (e.g., for specific GPUs), though the default often covers many common ones:*
        `cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"`

    *   **Enable OpenCL (and disable CUDA if desired):**
        ```bash
        cmake .. -DBUILD_OPENCL=ON -DBUILD_CUDA=OFF
        ```

    *   **Build Shared Library (required for C API and language bindings):**
        ```bash
        cmake .. -DBUILD_SHARED_LIB=ON
        ```

    *   **Build Python Bindings (implies `BUILD_SHARED_LIB=ON`):**
        ```bash
        cmake .. -DBUILD_PYTHON_BINDINGS=ON
        # CMake will attempt to find Python and will download pybind11 via FetchContent.
        # You might need to guide CMake to your Python installation if it's not found:
        # cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=/path/to/python
        ```
    *   **Building Rust Bindings (requires `BUILD_SHARED_LIB=ON` and Rust toolchain):**
        The C shared library (`libbitcrack_shared`) must be built and installed or its location made known to Cargo first. Then, navigate to the `bindings/rust/bitcrack_rs_bindings` directory and run `cargo build`. See `bindings/rust/README.md` for more detailed instructions if available.

3.  **Compile the Project:**

    *   **Linux / macOS:**
        ```bash
        cmake --build . -- -j$(nproc)  # Uses make by default, builds in parallel
        # Or simply:
        # make -j$(nproc)
        ```
    *   **Windows (using Visual Studio as generator):**
        Open the `BitCrack.sln` file generated in the `build` directory with Visual Studio and build the desired targets (e.g., `cuBitCrack`, `bitcrack_shared`).
        Or, use CMake to build from the command line (ensure your environment is set up for MSVC, e.g., by using a Developer Command Prompt):
        ```bash
        cmake --build . --config Release --target cuBitCrack
        # To build all:
        # cmake --build . --config Release
        ```

### Installation (Optional)

After building, you can install the compiled components (executables, libraries, headers) to a specified location using:
```bash
cmake --install . --prefix /path/to/your/install_directory
# On Linux, common prefix might be /usr/local or ~/.local
# On Windows, e.g., C:/Program Files/BitCrack
```
If no prefix is given, it might install to a system default (e.g., `/usr/local` on Linux).
This is particularly useful for making `libbitcrack_shared` and `bitcrack_capi.h` available for the Rust bindings or other projects.
The Python module also has install rules to place it into a Python `site-packages` directory.

## Running BitCrack

The main command-line interface (CLI) executable is `cuBitCrack` (for CUDA devices) or `clBitCrack` (for OpenCL devices), typically found in the `build/KeyFinder/Release` directory after compilation, or in your system path if installed.

### Command-Line Options

Many aspects of BitCrack's operation can be controlled via command-line options. Here are some key ones, including recent additions:

*   `-i, --in FILE`: Read target addresses from FILE (one address per line). Use "-" for stdin.
*   `-o, --out FILE`: Append found private keys to FILE.
*   `-d, --device IDs`: Specify one or more comma-separated **CUDA device IDs** to use (e.g., `-d 0` or `-d 0,1,2`). If not specified, defaults to device 0. (OpenCL device selection might still use a single ID or require specific OpenCL platform/device indexing).
*   `--keyspace KEYSPACE`: Define the range of private keys to search (e.g., `"START:END"` or `"START:+COUNT"`).
*   `--continue FILE`: Save progress to FILE and resume from it. Essential for long runs. Supports multi-GPU progress.
*   `--checkpoint-interval SECONDS`: Interval in seconds to write the checkpoint file (default is 60).
*   `--compression MODE`: `compressed`, `uncompressed`, or `both` (default: `compressed`).
*   `-c, -u`: Shortcuts for compressed/uncompressed modes.
*   `--list-devices`: Display available computing devices.
*   `--stride N`: Key increment step.
*   `--blocks N, -t THREADS, -p POINTS_PER_THREAD`: Manual GPU parameters. If `N`, `THREADS`, or `POINTS_PER_THREAD` are set to 0 (or not specified), BitCrack will attempt to auto-tune them for the selected CUDA device(s).
*   `--logfile FILEPATH`: Redirect detailed log output to the specified file.
*   `--loglevel LEVEL`: Set logging verbosity. Options: `trace`, `debug`, `info`, `warn`, `error`, `critical`, `off`. Default is `info`.

**Example CLI Usage:**

*   **Search for a single address on default CUDA device (GPU 0), auto-tuning GPU parameters:**
    ```bash
    ./cuBitCrack "1SomeBitcoinAddress..."
    ```
*   **Search for addresses from a file on multiple CUDA GPUs (0 and 1), starting from a specific key, with a 5-minute checkpoint interval, and verbose debug logging to a file:**
    ```bash
    ./cuBitCrack -d 0,1 --keyspace "20000000000000000:3FFFFFFFFFFFFFFFF" --in addresses.txt --continue progress.dat --checkpoint-interval 300 --loglevel debug --logfile bitcrack_run.log
    ```

### Using BitCrack as a Library

BitCrack can also be used as a library from Python and Rust.

#### Python Example

Ensure you have built the Python bindings (`-DBUILD_PYTHON_BINDINGS=ON`) and the `bitcrack_python` module is in your `PYTHONPATH` or installed.

```python
import bitcrack_python as bc
import time

print(f"BitCrack Python module loaded.")

# Create a session
session = bc.create_session()
if session == 0:
    print("Error creating BitCrack session")
    exit()

try:
    # Configure
    # Use device 0, auto-config for blocks/threads/points (0,0,0)
    bc.set_device(session, 0, 0, 0, 0) # Errors will raise RuntimeError
    print("Device set.")

    # Set a small keyspace for example: 1 to FFFFFFFF
    bc.set_keyspace(session, "1", "FFFFFFFF")
    print("Keyspace set.")

    # Add target addresses
    targets = ["1BitcoinEaterAddressDontSendf59kuE", "1AnotherAddress..."] # Replace with actual test addresses
    bc.add_targets(session, targets)
    print(f"Targets added: {targets}")

    # Set compression mode (0: compressed, 1: uncompressed, 2: both)
    # Constants are exposed: bc.COMPRESSION_COMPRESSED, bc.COMPRESSION_UNCOMPRESSED, bc.COMPRESSION_BOTH
    bc.set_compression_mode(session, bc.COMPRESSION_COMPRESSED)
    print("Compression mode set.")

    # Initialize the search (must be called before starting)
    bc.init_search(session)
    print("Search initialized.")

    # Start search asynchronously
    print("Starting search...")
    bc.start_search_async(session)

    while bc.is_search_running(session):
        print("Search running... polling for results in 5s...")
        time.sleep(5)

        try:
            # Poll for up to 10 results
            found_keys = bc.poll_results(session, 10)
            if found_keys:
                for key_info in found_keys:
                    print("\n*** Key Found! ***")
                    print(f"  Address: {key_info.address_base58}")
                    print(f"  Private Key (hex): {key_info.private_key_hex}")
                    print(f"  Compressed: {'Yes' if key_info.is_compressed else 'No'}")
                    print(f"  Public Key: {key_info.public_key_hex}")

                    print("Stopping search as key was found in example.")
                    bc.stop_search(session) # Signal search to stop
                    break
            if not bc.is_search_running(session): # Check if stop_search took effect
                 break
        except RuntimeError as poll_err:
            print(f"Error polling results: {poll_err}")
            bc.stop_search(session) # Stop on error
            break

    print("Search stopped or completed.")

except RuntimeError as e:
    print(f"A BitCrack operation failed: {e}")
except Exception as ex:
    print(f"An unexpected Python error occurred: {ex}")
finally:
    if session != 0:
        print("Destroying session...")
        bc.destroy_session(session)
        print("Session destroyed.")
```

#### Rust Example

Ensure `libbitcrack_shared` is built and accessible (e.g., in `LD_LIBRARY_PATH` or installed system-wide). Then build and run your Rust project that depends on the `bitcrack_rs_bindings` crate.

```rust
// In your Rust project's main.rs or example file
// Add to Cargo.toml: bitcrack_rs_bindings = { path = "/path/to/bitcrack_rs_bindings_crate" } or from crates.io if published

use bitcrack_rs_bindings::{BitCrack, FoundKey, COMPRESSION_COMPRESSED}; // Adjust path/name if needed
use std::{thread, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("BitCrack Rust Bindings Example");

    let bc_session = BitCrack::new().map_err(|e| e.to_string())?;
    println!("Session created.");

    bc_session.set_device(0, 0, 0, 0).map_err(|e| e.to_string())?; // Device 0, auto params
    println!("Device set.");

    bc_session.set_keyspace("1", "FFFFFFFF").map_err(|e| e.to_string())?; // Small range
    println!("Keyspace set.");

    let targets = vec!["1BitcoinEaterAddressDontSendf59kuE", "1AnotherAddress..."]; // Replace
    bc_session.add_targets(targets).map_err(|e| e.to_string())?;
    println!("Targets added.");

    bc_session.set_compression_mode(COMPRESSION_COMPRESSED).map_err(|e| e.to_string())?;
    println!("Compression mode set.");

    bc_session.init_search().map_err(|e| e.to_string())?;
    println!("Search initialized.");

    println!("Starting search...");
    bc_session.start_search_async().map_err(|e| e.to_string())?;

    loop {
        if !bc_session.is_search_running() {
            break;
        }
        println!("Search running... polling for results in 5s...");
        thread::sleep(Duration::from_secs(5));

        match bc_session.poll_results(10) {
            Ok(keys) => {
                if !keys.is_empty() {
                    for key_info in keys {
                        println!("
*** Key Found! ***");
                        println!("  Address: {}", key_info.address_base58);
                        println!("  Private Key (hex): {}", key_info.private_key_hex);
                        println!("  Compressed: {}", if key_info.is_compressed { "Yes" } else { "No" });
                        println!("  Public Key: {}", key_info.public_key_hex);

                        println!("Key found, stopping search.");
                        bc_session.stop_search(); // Signal search to stop
                        break; // out of inner loop
                    }
                }
            }
            Err(e) => {
                eprintln!("Error polling results: {}", e);
                bc_session.stop_search(); // Stop on error
                break; // out of outer loop
            }
        }
    }

    println!("Search stopped or completed.");
    // Session automatically destroyed when bc_session goes out of scope due to Drop trait implementation
    Ok(())
}
```

### Choosing the right parameters for your device

GPUs have many cores. Work for the cores is divided into blocks. Each block contains threads.

There are 3 parameters that affect performance: blocks, threads per block, and keys per thread.


`blocks:` Should be a multiple of the number of compute units on the device. The default is 32.

`threads:` The number of threads in a block. This must be a multiple of 32. The default is 256.

`Keys per thread:` The number of keys each thread will process. The performance (keys per second)
increases asymptotically with this value. The default is256. Increasing this value will cause the
kernel to run longer, but more keys will be processed.

### Supporting this project

If you find this project useful and would like to support it, consider making a donation. Your support is greatly appreciated!

**BTC**: `1LqJ9cHPKxPXDRia4tteTJdLXnisnfHsof`

**LTC**: `LfwqkJY7YDYQWqgR26cg2T1F38YyojD67J`

**ETH**: `0xd28082CD48E1B279425346E8f6C651C45A9023c5`

### Contact

Send any questions or comments to bitcrack.project@gmail.com

<hr/>
<a name="chinese-version"></a>
## 中文版
# BitCrack

一款用于暴力破解比特币私钥的工具。本项目的主要目的是为解决著名的【比特币谜题交易】（[Bitcoin puzzle transaction](https://blockchain.info/tx/08389f34c98c606322740c0be6a7125d9860bb8d5cb182c02f98461e5fa6cd15)）贡献力量：该交易包含32个地址，破解难度依次递增。

## 主要特性与优化 (现代化改造后)

此版本的 BitCrack 包含大量增强功能:

*   **现代化构建系统**: 使用 CMake (版本 3.18+) 进行跨平台编译, 支持现代 C++17 编译器, 并简化了依赖管理 (例如, 自动获取 `spdlog`, `pybind11`).
*   **增强的 GPU 支持**:
    *   默认支持更新的 CUDA 架构 (例如 Ampere, Lovelace, Hopper).
    *   根据检测到的 GPU 能力动态调整 CUDA 内核启动参数 (线程块数、线程数), 以优化开箱即用的性能.
*   **高效的 ECC 运算**: 在 CPU (用于工具函数) 和 GPU (用于高性能公钥生成) 上均实现了针对 `k*G` 的固定窗口 ECC 点乘优化, 显著加快了此核心加密步骤的速度.
*   **改进的地址处理**:
    *   强大的 Base58Check 解码功能, 能够正确解析各种比特币地址类型 (例如, 以 '1' 开头的 P2PKH 地址和以 '3' 开头的 P2SH 地址) 并提取其 HASH160.
    *   CPU 端的目标校验 (在 GPU Bloom Filter之后) 使用 `std::unordered_set`, 在处理大量目标地址 (百万级别) 时效率很高.
*   **高级性能优化**:
    *   **CUDA Streams**: 用于实现 GPU 内核执行、设备到主机的数据传输 (结果) 以及 CPU 端结果处理的流水线作业, 实现了更好的操作重叠并提高了吞吐量.
    *   **多 GPU 支持**: 允许通过命令行指定多个 CUDA 设备来分配工作负载, 从而显著提高整体密钥搜索速率.
*   **跨语言兼容性与可扩展性**:
    *   **共享库**: BitCrack 核心逻辑可被构建为共享库 (`libbitcrack_shared.so` 或 `bitcrack_shared.dll`).
    *   **C API**: 提供了 C 语言 API (`bitcrack_capi.h`), 暴露核心功能 (会话管理、配置、异步搜索、结果轮询), 便于集成到其他软件.
    *   **Python 绑定**: 使用 `pybind11` 构建了 Python 模块 (`bitcrack_python`), 允许从 Python 脚本直接控制 BitCrack 并获取结果.
    *   **Rust 绑定**: 使用 `bindgen` 构建了 Rust crate (`bitcrack_rs_bindings`), 在 C API 之上提供了安全的 Rust 封装.
*   **改进的日志与恢复功能**:
    *   集成了 `spdlog` 以实现结构化、可分级的日志记录 (到控制台, 并可选地记录到文件). 日志级别 (trace, debug, info, warn, error, critical, off) 和日志文件路径可通过命令行配置.
    *   强大的多 GPU 感知检查点功能, 能够为每个活动的 GPU 独立保存和恢复进度, 允许扫描任务断点续传. 检查点间隔可配置.

## 安装说明

### 系统必备条件

*   **CMake**: 版本 3.18 或更高. ([下载 CMake](https://cmake.org/download/))
*   **C++ 编译器**: 支持 C++17 的编译器 (例如, Windows 上的 MSVC, Linux 上的 GCC 或 Clang).
*   **CUDA Toolkit**: 版本 10.1 或更新版本 (用于 CUDA 支持). 包含 NVIDIA GPU 驱动和 `nvcc` 编译器. ([下载 CUDA Toolkit](https://developer.nvidia.com/cuda-downloads))
*   **OpenCL SDK (可选)**: 如果您计划构建或使用 OpenCL 版本 (`clBitCrack`). 通常可以在 GPU 供应商的驱动程序包 (AMD, Intel, NVIDIA) 中找到, 或有时包含在 CUDA Toolkit 中.
*   **Python (可选, 用于 Python 绑定)**: Python 3.6 或更高版本, 包括开发头文件 (例如, Debian/Ubuntu 上的 `python3-dev`). Pip 也常用于 Python 包管理.
*   **Rust (可选, 用于 Rust 绑定)**: Rust 工具链 (rustc, cargo). ([安装 Rust](https://www.rust-lang.org/tools/install))
*   **Git**: 用于克隆代码仓库.

### 使用 CMake 构建 (推荐方法)

1.  **克隆代码仓库:**
    ```bash
    git clone https://github.com/brichard19/BitCrack.git # 或您的 fork/源码位置
    cd BitCrack
    ```

2.  **使用 CMake 配置项目:**
    最佳实践是创建一个独立的构建目录.
    ```bash
    mkdir build
    cd build
    ```

    运行 CMake 配置项目。以下是一些常见的配置示例:

    *   **默认 (CUDA, Release 模式):**
        ```bash
        cmake ..
        # 指定构建类型 (Release 或 Debug):
        # cmake -DCMAKE_BUILD_TYPE=Release ..
        ```
        *您可以指定要为其构建的 CUDA 架构 (例如, 针对特定 GPU), 尽管默认设置通常涵盖许多常见架构:*
        `cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"`

    *   **启用 OpenCL (如果需要, 可禁用 CUDA):**
        ```bash
        cmake .. -DBUILD_OPENCL=ON -DBUILD_CUDA=OFF
        ```

    *   **构建共享库 (C API 和语言绑定所需):**
        ```bash
        cmake .. -DBUILD_SHARED_LIB=ON
        ```

    *   **构建 Python 绑定 (意味着 `BUILD_SHARED_LIB=ON`):**
        ```bash
        cmake .. -DBUILD_PYTHON_BINDINGS=ON
        # CMake 将尝试查找 Python, 并通过 FetchContent 下载 pybind11.
        # 如果 CMake 未能自动找到您的 Python 安装, 您可能需要指定路径:
        # cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=/path/to/python
        ```
    *   **构建 Rust 绑定 (需要 `BUILD_SHARED_LIB=ON` 和 Rust 工具链):**
        C 共享库 (`libbitcrack_shared`) 必须首先被构建并安装, 或者将其位置告知 Cargo. 然后, 导航到 `bindings/rust/bitcrack_rs_bindings` 目录并运行 `cargo build`. 如果可用, 请参阅 `bindings/rust/README.md` 获取更详细的说明.

3.  **编译项目:**

    *   **Linux / macOS:**
        ```bash
        cmake --build . -- -j$(nproc)  # 默认使用 make, 并行构建
        # 或者简单地:
        # make -j$(nproc)
        ```
    *   **Windows (使用 Visual Studio 作为生成器):**
        使用 Visual Studio 打开在 `build` 目录中生成的 `BitCrack.sln` 解决方案文件, 然后构建所需的目标 (例如, `cuBitCrack`, `bitcrack_shared`).
        或者, 从命令行使用 CMake 进行构建 (确保您的环境已为 MSVC 设置好, 例如, 使用开发者命令提示符):
        ```bash
        cmake --build . --config Release --target cuBitCrack
        # 构建所有目标:
        # cmake --build . --config Release
        ```

### 安装 (可选)

构建完成后, 您可以使用以下命令将编译好的组件 (可执行文件、库、头文件) 安装到指定位置:
```bash
cmake --install . --prefix /path/to/your/install_directory
# 在 Linux 上, 常用的前缀可能是 /usr/local 或 ~/.local
# 在 Windows 上, 例如 C:/Program Files/BitCrack
```
如果未指定前缀, 它可能会安装到系统默认位置 (例如 Linux 上的 `/usr/local`).
这对于使 `libbitcrack_shared` 和 `bitcrack_capi.h` 可用于 Rust 绑定或其他项目特别有用.
Python 模块也有安装规则, 会将其放置到 Python `site-packages` 目录中.

## 运行 BitCrack

主命令行界面 (CLI) 可执行文件是 `cuBitCrack` (适用于 CUDA 设备), 或 `clBitCrack` (适用于 OpenCL 设备), 编译后通常位于 `build/KeyFinder/Release` 目录中, 如果已安装则可能在您的系统路径中.

### 命令行选项

BitCrack 的许多操作方面都可以通过命令行选项进行控制. 以下是一些关键选项, 包括最近新增的:

*   `-i, --in FILE`: 从 FILE 文件中读取目标地址 (每行一个地址). 使用 "-" 代表标准输入.
*   `-o, --out FILE`: 将找到的私钥追加到 FILE 文件中.
*   `-d, --device IDs`: 指定一个或多个用逗号分隔的 **CUDA 设备 ID** (例如, `-d 0` 或 `-d 0,1,2`). 如果未指定, 则默认为设备 0. (OpenCL 设备选择可能仍使用单个 ID 或需要特定的 OpenCL 平台/设备索引).
*   `--keyspace KEYSPACE`: 定义要搜索的私钥范围 (例如, `"START:END"` 或 `"START:+COUNT"`).
*   `--continue FILE`: 将进度保存到 FILE 文件并从中恢复. 对长时间运行至关重要. 支持多 GPU 进度.
*   `--checkpoint-interval SECONDS`: 写入检查点文件的时间间隔 (秒, 默认为 60).
*   `--compression MODE`: `compressed`, `uncompressed`, 或 `both` (默认: `compressed`).
*   `-c, -u`: 压缩/未压缩模式的快捷方式.
*   `--list-devices`: 显示可用的计算设备.
*   `--stride N`: 密钥递增步长.
*   `--blocks N, -t THREADS, -p POINTS_PER_THREAD`: 手动 GPU 参数. 如果 `N`, `THREADS`, 或 `POINTS_PER_THREAD` 设置为 0 (或未指定), BitCrack 将尝试为选定的 CUDA 设备自动调整它们.
*   `--logfile FILEPATH`: 将详细的日志输出重定向到指定文件.
*   `--loglevel LEVEL`: 设置日志详细级别. 选项: `trace`, `debug`, `info`, `warn`, `error`, `critical`, `off` (默认: `info`).

**CLI 使用示例:**

*   **在默认 CUDA 设备 (GPU 0) 上搜索单个地址, 自动调整 GPU 参数:**
    ```bash
    ./cuBitCrack "1SomeBitcoinAddress..."
    ```
*   **在多个 CUDA GPU (0 和 1) 上从文件中搜索地址, 从特定密钥开始, 检查点间隔为5分钟, 并将详细的调试日志记录到文件:**
    ```bash
    ./cuBitCrack -d 0,1 --keyspace "20000000000000000:3FFFFFFFFFFFFFFFF" --in addresses.txt --continue progress.dat --checkpoint-interval 300 --loglevel debug --logfile bitcrack_run.log
    ```

### 作为库使用 BitCrack

BitCrack 也可以作为库从 Python 和 Rust 中使用.

#### Python 示例

确保您已构建 Python 绑定 (`-DBUILD_PYTHON_BINDINGS=ON`) 并且 `bitcrack_python` 模块位于您的 `PYTHONPATH` 中或已安装.

```python
import bitcrack_python as bc
import time

print(f"BitCrack Python 模块已加载.")

# 创建一个会话
session = bc.create_session()
if session == 0:
    print("创建 BitCrack 会话失败")
    exit()

try:
    # 配置: 设备 0, 自动配置 blocks/threads/points (0,0,0)
    bc.set_device(session, 0, 0, 0, 0) # 错误将引发 RuntimeError
    print("设备已设置.")

    # 设置一个小的密钥空间作为示例: 1 到 FFFFFFFF
    bc.set_keyspace(session, "1", "FFFFFFFF")
    print("密钥空间已设置.")

    # 添加目标地址
    targets = ["1BitcoinEaterAddressDontSendf59kuE", "1AnotherAddress..."] # 替换为实际测试地址
    bc.add_targets(session, targets)
    print(f"目标已添加: {targets}")

    # 设置压缩模式 (0: compressed, 1: uncompressed, 2: both)
    # 常量已公开: bc.COMPRESSION_COMPRESSED, bc.COMPRESSION_UNCOMPRESSED, bc.COMPRESSION_BOTH
    bc.set_compression_mode(session, bc.COMPRESSION_COMPRESSED)
    print("压缩模式已设置.")

    # 初始化搜索 (必须在开始前调用)
    bc.init_search(session)
    print("搜索已初始化.")

    # 异步启动搜索
    print("开始搜索...")
    bc.start_search_async(session)

    while bc.is_search_running(session):
        print("搜索正在运行... 5秒后轮询结果...")
        time.sleep(5)

        try:
            # 轮询最多 10 个结果
            found_keys = bc.poll_results(session, 10)
            if found_keys:
                for key_info in found_keys:
                    print("\n*** 找到密钥! ***")
                    print(f"  地址: {key_info.address_base58}")
                    print(f"  私钥 (十六进制): {key_info.private_key_hex}")
                    print(f"  是否压缩: {'是' if key_info.is_compressed else '否'}")
                    print(f"  公钥: {key_info.public_key_hex}")

                    print("示例中找到密钥, 停止搜索.")
                    bc.stop_search(session) # 通知搜索停止
                    break
            if not bc.is_search_running(session): # 检查 stop_search 是否生效
                 break
        except RuntimeError as poll_err:
            print(f"轮询结果时出错: {poll_err}")
            bc.stop_search(session) # 出错时停止
            break

    print("搜索已停止或完成.")

except RuntimeError as e:
    print(f"BitCrack 操作失败: {e}")
except Exception as ex:
    print(f"发生意外的 Python 错误: {ex}")
finally:
    if session != 0:
        print("销毁会话...")
        bc.destroy_session(session)
        print("会话已销毁.")
```

#### Rust 示例

确保 `libbitcrack_shared` 已构建且可访问 (例如, 在 `LD_LIBRARY_PATH` 中或已系统范围安装). 然后构建并运行依赖于 `bitcrack_rs_bindings` crate 的 Rust 项目.

```rust
// In your Rust project's main.rs or example file
// Add to Cargo.toml: bitcrack_rs_bindings = { path = "/path/to/bitcrack_rs_bindings_crate" } or from crates.io if published

use bitcrack_rs_bindings::{BitCrack, FoundKey, COMPRESSION_COMPRESSED}; // Adjust path/name if needed
use std::{thread, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("BitCrack Rust 绑定示例");

    let bc_session = BitCrack::new().map_err(|e| e.to_string())?;
    println!("会话已创建.");

    bc_session.set_device(0, 0, 0, 0).map_err(|e| e.to_string())?; // 设备 0, 自动参数
    println!("设备已设置.");

    bc_session.set_keyspace("1", "FFFFFFFF").map_err(|e| e.to_string())?; // 小范围
    println!("密钥空间已设置.");

    let targets = vec!["1BitcoinEaterAddressDontSendf59kuE", "1AnotherAddress..."]; // 替换
    bc_session.add_targets(targets).map_err(|e| e.to_string())?;
    println!("目标已添加.");

    bc_session.set_compression_mode(COMPRESSION_COMPRESSED).map_err(|e| e.to_string())?;
    println!("压缩模式已设置.");

    bc_session.init_search().map_err(|e| e.to_string())?;
    println!("搜索已初始化.");

    println!("开始搜索...");
    bc_session.start_search_async().map_err(|e| e.to_string())?;

    loop {
        if !bc_session.is_search_running() {
            break;
        }
        println!("搜索正在运行... 5秒后轮询结果...");
        thread::sleep(Duration::from_secs(5));

        match bc_session.poll_results(10) {
            Ok(keys) => {
                if !keys.is_empty() {
                    for key_info in keys {
                        println!("
*** 找到密钥! ***");
                        println!("  地址: {}", key_info.address_base58);
                        println!("  私钥 (十六进制): {}", key_info.private_key_hex);
                        println!("  是否压缩: {}", if key_info.is_compressed { "是" } else { "否" });
                        println!("  公钥: {}", key_info.public_key_hex);

                        println!("示例中找到密钥, 停止搜索.");
                        bc_session.stop_search(); // 通知搜索停止
                        break; // 跳出内部循环
                    }
                }
            }
            Err(e) => {
                eprintln!("轮询结果时出错: {}", e);
                bc_session.stop_search(); // 出错时停止
                break; // 跳出外部循环
            }
        }
    }

    println!("搜索已停止或完成.");
    // 由于 Drop trait 的实现, bc_session 在离开作用域时会自动销毁会话
    Ok(())
}
```

### 为您的设备选择正确的参数

GPU 包含许多核心。核心的工作被划分为块（blocks），每个块包含线程（threads）。

有3个影响性能的参数：块数（blocks）、每块线程数（threads per block）和每线程密钥数（keys per thread）。

`blocks:` 应为设备上计算单元数量的倍数。默认值为32。

`threads:` 一个块中的线程数。必须是32的倍数。默认值为256。

`Keys per thread:` 每个线程将处理的密钥数量。性能（密钥/秒）随此值渐近增加。默认值为256。增加此值将导致内核运行时间更长，但会处理更多密钥。

### 支持本项目

如果您觉得这个项目有用并希望支持它，请考虑捐赠。非常感谢您的支持！

**BTC**: `1LqJ9cHPKxPXDRia4tteTJdLXnisnfHsof`

**LTC**: `LfwqkJY7YDYQWqgR26cg2T1F38YyojD67J`

**ETH**: `0xd28082CD48E1B279425346E8f6C651C45A9023c5`

### 联系方式

如有任何问题或意见，请发送邮件至 bitcrack.project@gmail.com
