#include "CudaAtomicList.h"
#include "CudaAtomicList.cuh"

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

static __constant__ void *_LIST_BUF[1];
static __constant__ unsigned int *_LIST_SIZE[1];


__device__ void atomicListAdd(void *info, unsigned int size)
{
	unsigned int count = atomicAdd(_LIST_SIZE[0], 1);

	unsigned char *ptr = (unsigned char *)(_LIST_BUF[0]) + count * size;

	memcpy(ptr, info, size);
}

static cudaError_t setListPtr(void *ptr, unsigned int *numResults)
{
	cudaError_t err = cudaMemcpyToSymbol(_LIST_BUF, &ptr, sizeof(void *));

	if(err) {
		return err;
	}

	err = cudaMemcpyToSymbol(_LIST_SIZE, &numResults, sizeof(unsigned int *));

	return err;
}


cudaError_t CudaAtomicList::init(unsigned int itemSize, unsigned int maxItems)
{
	_itemSize = itemSize;

	// The number of results found in the most recent kernel run
	_countHostPtr = NULL;
	cudaError_t err = cudaHostAlloc(&_countHostPtr, sizeof(unsigned int), cudaHostAllocMapped);
	if(err) {
		goto end;
	}

	// Number of items in the list
	_countDevPtr = NULL;
	err = cudaHostGetDevicePointer(&_countDevPtr, _countHostPtr, 0);
	if(err) {
		goto end;
	}
	*_countHostPtr = 0;

	// Storage for results data
	_hostPtr = NULL;
	err = cudaHostAlloc(&_hostPtr, itemSize * maxItems, cudaHostAllocMapped);
	if(err) {
		goto end;
	}

	// Storage for results data (device to host pointer)
	_devPtr = NULL;
	err = cudaHostGetDevicePointer(&_devPtr, _hostPtr, 0);

	if(err) {
		goto end;
	}

	err = setListPtr(_devPtr, _countDevPtr);

end:
	if(err) {
		cudaFreeHost(_countHostPtr);

		cudaFree(_countDevPtr);

		cudaFreeHost(_hostPtr);

		cudaFree(_devPtr);
	}

	return err;
}

unsigned int CudaAtomicList::size()
{
	return *_countHostPtr;
}

void CudaAtomicList::clear()
{
	*_countHostPtr = 0;
}

unsigned int CudaAtomicList::read(void *ptr, unsigned int count)
{
	if(count >= *_countHostPtr) {
		count = *_countHostPtr;
	}

	memcpy(ptr, _hostPtr, count * _itemSize);

	return count;
}

// New method to get current item count by synchronous DtoH copy of the count
int CudaAtomicList::getCurrentItemCount()
{
    if (!_countDevPtr) return 0; // Not initialized

    unsigned int host_count = 0;
    // Synchronously copy the count from device memory to host.
    // _countDevPtr is the device pointer to the count.
    cudaError_t err = cudaMemcpy(&host_count, _countDevPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        // Handle error, e.g., throw or log. For now, return 0 or last known good.
        // This could also be an assertion in debug builds.
        // Returning 0 might be misleading if copy fails partially.
        // Consider throwing an exception or returning -1 to indicate error.
        fprintf(stderr, "CudaAtomicList::getCurrentItemCount: cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err));
        return 0; // Or throw
    }
    return static_cast<int>(host_count);
}

// New asynchronous read method
// listSizeInBytes should be itemCount * _itemSize
void CudaAtomicList::readAsync(unsigned char *dest, int listSizeInBytes, cudaStream_t stream)
{
    if (!_devPtr || listSizeInBytes == 0) {
        // Nothing to copy or not initialized
        return;
    }

    // _devPtr is the device pointer to the list data.
    cudaError_t err = cudaMemcpyAsync(dest, _devPtr, listSizeInBytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        // Handle error. This should ideally throw or be checked by the caller.
        // For now, printing error.
        fprintf(stderr, "CudaAtomicList::readAsync: cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
        // If an exception mechanism is available, use it: throw CudaException(err);
    }
    // Note: This function does not synchronize the stream. The caller is responsible.
}

void CudaAtomicList::cleanup()
{
	cudaFreeHost(_countHostPtr);

	cudaFree(_countDevPtr);

	cudaFreeHost(_hostPtr);

	cudaFree(_devPtr);
}