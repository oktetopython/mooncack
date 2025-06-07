#ifndef _EC_H
#define _EC_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include "secp256k1.h"


class CudaDeviceKeys {

private:
	int _blocks;

	int _threads;

	int _pointsPerThread;

	unsigned int _numKeys;

	unsigned int *_devX;

	unsigned int *_devY;

	unsigned int *_devPrivate;

	// unsigned int *_devChain; // Removed

	// unsigned int *_devBasePointX; // Already Removed
	// unsigned int *_devBasePointY; // Already Removed
	// int _step; // Already Removed

	int getIndex(int block, int thread, int idx);

	void splatBigInt(unsigned int *dest, int block, int thread, int idx, const secp256k1::uint256 &i);

	secp256k1::uint256 readBigInt(unsigned int *src, int block, int thread, int idx);

	// cudaError_t allocateChainBuf(unsigned int count); // Removed

	cudaError_t initializePublicKeys(size_t count);

	// cudaError_t initializeBasePoints(); // Removed


public:

	CudaDeviceKeys()
	{
		_numKeys = 0;
		_devX = NULL;
		_devY = NULL;
		_devPrivate = NULL;
		// _devChain = NULL; // Removed
		// _devBasePointX = NULL; // Already Removed
		// _devBasePointY = NULL; // Already Removed
		// _step = 0; // Already Removed
	}

	~CudaDeviceKeys()
	{
		clearPublicKeys();
		clearPrivateKeys();
	}

	cudaError_t init(int blocks, int threads, int pointsPerThread, const std::vector<secp256k1::uint256> &privateKeys);

	bool selfTest(const std::vector<secp256k1::uint256> &privateKeys);

	// cudaError_t doStep(); // Removed

	void clearPrivateKeys();

	void clearPublicKeys();

};

#endif