#ifndef _BRIDGE_H
#define _BRIDGE_H

#include<cuda.h>
#include<cuda_runtime.h>
#include<string>
#include "cudaUtil.h"
#include "secp256k1.h"


void callKeyFinderKernel(int blocks, int threads, int points, bool useDouble, int compression, cudaStream_t stream, cudaEvent_t eventToRecord);

// void waitForKernel(cudaStream_t stream); // Removed

cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y);
cudaError_t allocateChainBuf(unsigned int count);
void cleanupChainBuf();

#endif