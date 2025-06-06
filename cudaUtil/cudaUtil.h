#ifndef _CUDA_UTIL_H
#define _CUDA_UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace cuda {
	typedef struct {

		int id;
		int major;
		int minor;
		int mpCount;
		int cores; // Present, not explicitly requested to remove, will keep
		uint64_t mem; // Assuming this is totalGlobalMem
		std::string name;
		// New fields based on subtask
		int maxThreadsPerBlock;
		int maxThreadsPerMultiProcessor;
		int warpSize;
		size_t sharedMemPerBlock;
		// totalGlobalMem is already covered by 'mem'

	}CudaDeviceInfo;

	class CudaException
	{
	public:
		cudaError_t error;
		std::string msg;

		CudaException(cudaError_t err)
		{
			this->error = err;
			this->msg = std::string(cudaGetErrorString(err));
		}
	};

	CudaDeviceInfo getDeviceInfo(int device);

	std::vector<CudaDeviceInfo> getDevices();

	int getDeviceCount();
}
#endif