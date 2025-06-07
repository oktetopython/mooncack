#ifndef _ATOMIC_LIST_HOST_H
#define _ATOMIC_LIST_HOST_H

#include <cuda_runtime.h>

/**
 A list that multiple device threads can append items to. Items can be
 read and removed by the host
 */
class CudaAtomicList {

private:
	void *_devPtr;

	void *_hostPtr;

	unsigned int *_countHostPtr;

	unsigned int *_countDevPtr;

	unsigned int _maxSize;

	unsigned int _itemSize;

public:

	CudaAtomicList()
	{
		_devPtr = NULL;
		_hostPtr = NULL;
		_countHostPtr = NULL;
		_countDevPtr = NULL;
		_maxSize = 0;
		_itemSize = 0;
	}

	~CudaAtomicList()
	{
		cleanup();
	}

	cudaError_t init(unsigned int itemSize, unsigned int maxItems);

	unsigned int read(void *dest, unsigned int count); // Existing synchronous read

	// New asynchronous read method
	void readAsync(unsigned char *dest, int listSizeInBytes, cudaStream_t stream);

	// New method to get current item count by synchronous DtoH copy of the count
	int getCurrentItemCount();

	unsigned int size(); // This likely returns the host-side count or max size, need to check impl.

	void clear();

    void cleanup();

};

#endif