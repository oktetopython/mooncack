[English](#english-version) | [中文](#chinese-version)

<a name="english-version"></a>
## English Version
# BitCrack

A tool for brute-forcing Bitcoin private keys. The main purpose of this project is to contribute to the effort of solving the [Bitcoin puzzle transaction](https://blockchain.info/tx/08389f34c98c606322740c0be6a7125d9860bb8d5cb182c02f98461e5fa6cd15): A transaction with 32 addresses that become increasingly difficult to crack.


### Using BitCrack

#### Usage


Use `cuBitCrack.exe` for CUDA devices and `clBitCrack.exe` for OpenCL devices.

### Note: **clBitCrack.exe is still EXPERIMENTAL**, as users have reported critial bugs when running on some AMD and Intel devices.

**Note for Intel users:**

There is bug in Intel's OpenCL implementation which affects BitCrack. Details here: https://github.com/brichard19/BitCrack/issues/123


```
xxBitCrack.exe [OPTIONS] [TARGETS]

Where [TARGETS] are one or more Bitcoin address

Options:

-i, --in FILE
    Read addresses from FILE, one address per line. If FILE is "-" then stdin is read

-o, --out FILE
    Append private keys to FILE, one per line

-d, --device N
    Use device with ID equal to N

-b, --blocks BLOCKS
    The number of CUDA blocks

-t, --threads THREADS
    Threads per block

-p, --points NUMBER
    Each thread will process NUMBER keys at a time

--keyspace KEYSPACE
    Specify the range of keys to search, where KEYSPACE is in the format,

	START:END start at key START, end at key END
	START:+COUNT start at key START and end at key START + COUNT
    :END start at key 1 and end at key END
	:+COUNT start at key 1 and end at key 1 + COUNT

-c, --compressed
    Search for compressed keys (default). Can be used with -u to also search uncompressed keys

-u, --uncompressed
    Search for uncompressed keys, can be used with -c to search compressed keys

--compression MODE
    Specify the compression mode, where MODE is 'compressed' or 'uncompressed' or 'both'

--list-devices
    List available devices

--stride NUMBER
    Increment by NUMBER

--share M/N
    Divide the keyspace into N equal sized shares, process the Mth share

--continue FILE
    Save/load progress from FILE
```

#### Examples

The simplest usage, the keyspace will begin at 0, and the CUDA parameters will be chosen automatically
```
xxBitCrack.exe 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

Multiple keys can be searched at once with minimal impact to performance. Provide the keys on the command line, or in a file with one address per line
```
xxBitCrack.exe 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH 15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP 19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT
```

To start the search at a specific private key, use the `--keyspace` option:

```
xxBitCrack.exe --keyspace 766519C977831678F0000000000 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

The `--keyspace` option can also be used to search a specific range:

```
xxBitCrack.exe --keyspace 80000000:ffffffff 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

To periodically save progress, the `--continue` option can be used. This is useful for recovering
after an unexpected interruption:

```
xxBitCrack.exe --keyspace 80000000:ffffffff --continue progress.txt 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
...
GeForce GT 640   224/1024MB | 1 target 10.33 MKey/s (1,244,659,712 total) [00:01:58]
^C
xxBitCrack.exe --keyspace 80000000:ffffffff --continue progress.txt 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
...
GeForce GT 640   224/1024MB | 1 target 10.33 MKey/s (1,357,905,920 total) [00:02:12]
```


Use the `-b,` `-t` and `-p` options to specify the number of blocks, threads per block, and keys per thread.
```
xxBitCrack.exe -b 32 -t 256 -p 16 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

### Choosing the right parameters for your device

GPUs have many cores. Work for the cores is divided into blocks. Each block contains threads.

There are 3 parameters that affect performance: blocks, threads per block, and keys per thread.


`blocks:` Should be a multiple of the number of compute units on the device. The default is 32.

`threads:` The number of threads in a block. This must be a multiple of 32. The default is 256.

`Keys per thread:` The number of keys each thread will process. The performance (keys per second)
increases asymptotically with this value. The default is256. Increasing this value will cause the
kernel to run longer, but more keys will be processed.


### Build dependencies

Visual Studio 2019 (if on Windows)

For CUDA: CUDA Toolkit 10.1

For OpenCL: An OpenCL SDK (The CUDA toolkit contains an OpenCL SDK).


### Building in Windows

Open the Visual Studio solution.

Build the `clKeyFinder` project for an OpenCL build.

Build the `cuKeyFinder` project for a CUDA build.

Note: By default the NVIDIA OpenCL headers are used. You can set the header and library path for
OpenCL in the `BitCrack.props` property sheet.

### Building in Linux

Using `make`:

Build CUDA:
```
make BUILD_CUDA=1
```

Build OpenCL:
```
make BUILD_OPENCL=1
```

Or build both:
```
make BUILD_CUDA=1 BUILD_OPENCL=1
```

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

### 使用 BitCrack

#### 用法

对于 CUDA 设备，请使用 `cuBitCrack.exe`；对于 OpenCL 设备，请使用 `clBitCrack.exe`。

### 注意：**clBitCrack.exe 仍处于实验阶段**，已有用户报告在部分 AMD 和 Intel 设备上运行时存在严重错误。

**针对 Intel 用户的注意事项：**

Intel 的 OpenCL 实现中存在一个影响 BitCrack 的错误。详情请见：https://github.com/brichard19/BitCrack/issues/123

```
xxBitCrack.exe [选项] [目标地址]

其中 [目标地址] 是一个或多个比特币地址

选项：

-i, --in FILE
    从 FILE 文件中读取地址，每行一个地址。如果 FILE 是 "-" 则从标准输入读取

-o, --out FILE
    将找到的私钥追加到 FILE 文件中，每行一个

-d, --device N
    使用 ID 为 N 的设备

-b, --blocks BLOCKS
    CUDA 块的数量

-t, --threads THREADS
    每个块的线程数

-p, --points NUMBER
    每个线程一次处理 NUMBER 个密钥

--keyspace KEYSPACE
    指定要搜索的密钥范围，KEYSPACE 格式如下：

	START:END         从密钥 START 开始，到密钥 END 结束
	START:+COUNT      从密钥 START 开始，到密钥 START + COUNT 结束
    :END              从密钥 1 开始，到密钥 END 结束
	:+COUNT           从密钥 1 开始，到密钥 1 + COUNT 结束

-c, --compressed
    搜索压缩格式的密钥（默认）。可与 -u 结合使用以同时搜索未压缩密钥

-u, --uncompressed
    搜索未压缩格式的密钥，可与 -c 结合使用以同时搜索压缩密钥

--compression MODE
    指定压缩模式，MODE 可以是 'compressed'、'uncompressed' 或 'both'

--list-devices
    列出可用设备

--stride NUMBER
    密钥递增步长为 NUMBER

--share M/N
    将密钥空间分成 N 个等大的份额，处理第 M 个份额

--continue FILE
    从 FILE 文件中保存/加载进度
```

#### 示例

最简单的用法，密钥空间将从0开始，CUDA参数将自动选择：
```
xxBitCrack.exe 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

可以同时搜索多个密钥，对性能影响最小。可以在命令行中提供密钥，或者在一个文件中提供，每行一个地址：
```
xxBitCrack.exe 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH 15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP 19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT
```

要从特定的私钥开始搜索，请使用 `--keyspace` 选项：
```
xxBitCrack.exe --keyspace 766519C977831678F0000000000 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

`--keyspace` 选项也可以用于搜索特定范围：
```
xxBitCrack.exe --keyspace 80000000:ffffffff 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

要定期保存进度，可以使用 `--continue` 选项。这对于在意外中断后恢复非常有用：
```
xxBitCrack.exe --keyspace 80000000:ffffffff --continue progress.txt 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
...
GeForce GT 640   224/1024MB | 1 target 10.33 MKey/s (1,244,659,712 total) [00:01:58]
^C
xxBitCrack.exe --keyspace 80000000:ffffffff --continue progress.txt 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
...
GeForce GT 640   224/1024MB | 1 target 10.33 MKey/s (1,357,905,920 total) [00:02:12]
```

使用 `-b`、`-t` 和 `-p` 选项来指定块数、每块线程数和每线程密钥数：
```
xxBitCrack.exe -b 32 -t 256 -p 16 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

### 为您的设备选择正确的参数

GPU 包含许多核心。核心的工作被划分为块（blocks），每个块包含线程（threads）。

有3个影响性能的参数：块数（blocks）、每块线程数（threads per block）和每线程密钥数（keys per thread）。

`blocks:` 应为设备上计算单元数量的倍数。默认值为32。

`threads:` 一个块中的线程数。必须是32的倍数。默认值为256。

`Keys per thread:` 每个线程将处理的密钥数量。性能（密钥/秒）随此值渐近增加。默认值为256。增加此值将导致内核运行时间更长，但会处理更多密钥。

### 构建依赖

Visual Studio 2019 (如果在Windows上)

对于 CUDA: CUDA Toolkit 10.1

对于 OpenCL: OpenCL SDK (CUDA Toolkit 中包含 OpenCL SDK)。

### 在 Windows 上构建

打开 Visual Studio 解决方案。

为 OpenCL 构建，请生成 `clKeyFinder` 项目。

为 CUDA 构建，请生成 `cuKeyFinder` 项目。

注意：默认情况下使用 NVIDIA OpenCL 头文件。您可以在 `BitCrack.props` 属性表中设置 OpenCL 的头文件和库路径。

### 在 Linux 上构建

使用 `make`:

构建 CUDA 版本:
```
make BUILD_CUDA=1
```

构建 OpenCL 版本:
```
make BUILD_OPENCL=1
```

或者同时构建两者:
```
make BUILD_CUDA=1 BUILD_OPENCL=1
```

### 支持本项目

如果您觉得这个项目有用并希望支持它，请考虑捐赠。非常感谢您的支持！

**BTC**: `1LqJ9cHPKxPXDRia4tteTJdLXnisnfHsof`

**LTC**: `LfwqkJY7YDYQWqgR26cg2T1F38YyojD67J`

**ETH**: `0xd28082CD48E1B279425346E8f6C651C45A9023c5`

### 联系方式

如有任何问题或意见，请发送邮件至 bitcrack.project@gmail.com
