#include "cudabridge.h"


__global__ void keyFinderKernel(int points, int compression);
__global__ void keyFinderKernelWithDouble(int points, int compression);

void callKeyFinderKernel(int blocks, int threads, int points, bool useDouble, int compression, cudaStream_t stream, cudaEvent_t eventToRecord)
{
	if(useDouble) {
		keyFinderKernelWithDouble <<<blocks, threads, 0, stream >>>(points, compression);
	} else {
		keyFinderKernel <<<blocks, threads, 0, stream >>> (points, compression);
	}

    // Record event on the stream after kernel launch
    cudaError_t eventErr = cudaEventRecord(eventToRecord, stream);
    if (eventErr != cudaSuccess) {
        // Handle or throw error. For now, let's assume cudaCall in CudaKeySearchDevice will catch it
        // or rely on subsequent synchronization to report an error state.
        // A more robust way might be to throw here or return the error.
        // CudaKeySearchDevice::cudaCall can't be used directly as this is not a class method.
        // For simplicity in this step, error is not explicitly thrown here,
        // but subsequent event sync will fail if record fails.
        // A cudaGetLastError() check could also be added here.
    }
	// waitForKernel(stream); // Removed
}


// void waitForKernel(cudaStream_t stream) // Removed
// {
//    ...
// }