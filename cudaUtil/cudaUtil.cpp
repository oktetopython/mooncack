#include "cudaUtil.h"


cuda::CudaDeviceInfo cuda::getDeviceInfo(int device)
{
	cuda::CudaDeviceInfo devInfo;

	cudaDeviceProp properties;
	cudaError_t err = cudaSuccess;

	err = cudaSetDevice(device);

	if(err) {
		throw cuda::CudaException(err);
	}

	err = cudaGetDeviceProperties(&properties, device);
	
	if(err) {
		throw cuda::CudaException(err);
	}

	devInfo.id = device;
	devInfo.major = properties.major;
	devInfo.minor = properties.minor;
	devInfo.mpCount = properties.multiProcessorCount;
	devInfo.mem = properties.totalGlobalMem; // This is totalGlobalMem
	devInfo.name = std::string(properties.name);

	// Populate new fields
	devInfo.maxThreadsPerBlock = properties.maxThreadsPerBlock;
	devInfo.maxThreadsPerMultiProcessor = properties.maxThreadsPerMultiProcessor;
	devInfo.warpSize = properties.warpSize;
	devInfo.sharedMemPerBlock = properties.sharedMemPerBlock;

	int cores = 0;
	switch(devInfo.major) {
	case 1:
		cores = 8;
		break;
	case 2:
        if(devInfo.minor == 0) {
            cores = 32;
        } else {
            cores = 48;
        }
		break;
	case 3:
		cores = 192;
		break;
	case 5:
		cores = 128;
		break;
	case 6:
        if(devInfo.minor == 1 || devInfo.minor == 2) {
            cores = 128;
        } else {
            cores = 64;
        }
        break;
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
	devInfo.cores = cores;

	return devInfo;
}


std::vector<cuda::CudaDeviceInfo> cuda::getDevices()
{
	int count = getDeviceCount();

	std::vector<CudaDeviceInfo> devList;

	for(int device = 0; device < count; device++) {
		devList.push_back(getDeviceInfo(device));
	}

	return devList;
}

int cuda::getDeviceCount()
{
	int count = 0;

	cudaError_t err = cudaGetDeviceCount(&count);

    if(err) {
        throw cuda::CudaException(err);
    }

	return count;
}

