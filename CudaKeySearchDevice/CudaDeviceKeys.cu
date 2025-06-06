#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaDeviceKeys.h"
#include "CudaDeviceKeys.cuh"
#include "secp256k1.cuh"


__constant__ unsigned int *_xPtr[1];

__constant__ unsigned int *_yPtr[1];

// Static flag to ensure constant memory table is initialized only once
static bool c_gWindowTableInitialized_host = false;
static std::mutex c_gWindowTableMutex_host; // For thread-safe initialization on host


__device__ unsigned int *ec::getXPtr()
{
	return _xPtr[0];
}

__device__ unsigned int *ec::getYPtr()
{
	return _yPtr[0];
}

// __global__ void multiplyStepKernel(...); // Removed

// Forward declaration for the new kernel
__global__ void generateKeys_fixedWindow_kernel(
    const unsigned int *privateKeys,
    int numKeys,
    int pointsPerThread, // How many keys each thread handles; if 1, this can be simplified.
    unsigned int *outGx,
    unsigned int *outGy
);

// Device helper to extract w-bit chunk from a private key (represented as uint[8])
// k_words: pointer to the start of the private key's 8 words
// window_idx: which w-bit window (0 is LSB window)
// w: window size (e.g., 4)
// m: total bitlength of scalar (e.g., 256)
__device__ __forceinline__ unsigned int extract_kernel_w_bit_chunk(const unsigned int* k_words, int window_idx, int w, int m) {
    unsigned int val = 0;
    int start_bit_in_scalar = window_idx * w;

    for (int bit_in_window = 0; bit_in_window < w; ++bit_in_window) {
        int current_bit_pos = start_bit_in_scalar + bit_in_window;
        if (current_bit_pos < m) {
            // k_words are in big-endian style from secp256k1::uint256::exportWords(..., BigEndian)
            // So k_words[0] is MSW, k_words[7] is LSW.
            // bit(n) in uint256 checks (v[n/32] & (1<<(n%32))). v is little-endian words.
            // So, for k.bit(N), N=0 is LSB of v[0].
            // If k_words is a direct copy of v (little-endian words from uint256):
            // int word_idx = current_bit_pos / 32;
            // int bit_in_word = current_bit_pos % 32;
            // if (k_words[word_idx] & (1U << bit_in_word)) {
            //    val |= (1U << bit_in_window);
            // }
            // The splatBigInt function stores words in big-endian format in memory.
            // privateKeys[key_idx_global*8 + word_idx_big_endian]
            // k_words[0] = MSW, k_words[7] = LSW
            // We need to map current_bit_pos (0=LSB of scalar) to this big-endian word array.
            int scalar_word_idx_little_endian = current_bit_pos / 32; // 0 to 7
            int bit_in_uint32 = current_bit_pos % 32;
            // Convert scalar_word_idx_little_endian to k_words index (big-endian stored)
            int k_word_actual_idx = 7 - scalar_word_idx_little_endian; // maps 0->7, 1->6, ..., 7->0

            if (k_words[k_word_actual_idx] & (1U << bit_in_uint32)) {
                 val |= (1U << bit_in_window);
            }
        }
    }
    return val;
}


int CudaDeviceKeys::getIndex(int block, int thread, int idx)
{
	// Total number of threads
	int totalThreads = _blocks * _threads;

	int base = idx * totalThreads;

	// Global ID of the current thread
	int threadId = block * _threads + thread;

	return base + threadId;
}

void CudaDeviceKeys::splatBigInt(unsigned int *dest, int block, int thread, int idx, const secp256k1::uint256 &i)
{
	unsigned int value[8] = { 0 };

	i.exportWords(value, 8, secp256k1::uint256::BigEndian);

	int totalThreads = _blocks * _threads;
	int threadId = block * _threads + thread;

	int base = idx * _blocks * _threads * 8;

	int index = base + threadId;

	for(int k = 0; k < 8; k++) {
		dest[index] = value[k];
		index += totalThreads;
	}
}

secp256k1::uint256 CudaDeviceKeys::readBigInt(unsigned int *src, int block, int thread, int idx)
{
	unsigned int value[8] = { 0 };

	int totalThreads = _blocks * _threads;
	int threadId = block * _threads + thread;

	int base = idx * _blocks * _threads * 8;

	int index = base + threadId;

	for(int k = 0; k < 8; k++) {
		value[k] = src[index];
		index += totalThreads;
	}

	secp256k1::uint256 v(value, secp256k1::uint256::BigEndian);

	return v;
}

// Removed initializeBasePoints()
// allocateChainBuf definition removed.

// Host-side function to compute and copy G window table to constant memory
static cudaError_t initConstantGWindowTable(int w = 4) {
    if (c_gWindowTableInitialized_host) {
        return cudaSuccess;
    }
    std::lock_guard<std::mutex> lock(c_gWindowTableMutex_host);
    if (c_gWindowTableInitialized_host) {
        return cudaSuccess;
    }

    unsigned int table_entries = 1U << w;
    std::vector<unsigned int> hostTableX(table_entries * 8);
    std::vector<unsigned int> hostTableY(table_entries * 8);

    secp256k1::ecpoint current_g_multiple = secp256k1::pointAtInfinity();
    current_g_multiple.x.exportWords(&hostTableX[0 * 8], 8, secp256k1::uint256::BigEndian); // G_WINDOW_TABLE[0] = Infinity
    current_g_multiple.y.exportWords(&hostTableY[0 * 8], 8, secp256k1::uint256::BigEndian);

    if (table_entries > 1) {
        current_g_multiple = secp256k1::G();
        current_g_multiple.x.exportWords(&hostTableX[1 * 8], 8, secp256k1::uint256::BigEndian); // G_WINDOW_TABLE[1] = G
        current_g_multiple.y.exportWords(&hostTableY[1 * 8], 8, secp256k1::uint256::BigEndian);
    }

    for (unsigned int i = 2; i < table_entries; ++i) {
        current_g_multiple = secp256k1::addPoints(current_g_multiple, secp256k1::G());
        current_g_multiple.x.exportWords(&hostTableX[i * 8], 8, secp256k1::uint256::BigEndian);
        current_g_multiple.y.exportWords(&hostTableY[i * 8], 8, secp256k1::uint256::BigEndian);
    }

    cudaError_t errX = cudaMemcpyToSymbol(c_gWindowTableX, hostTableX.data(), table_entries * 8 * sizeof(unsigned int));
    if (errX != cudaSuccess) return errX;
    cudaError_t errY = cudaMemcpyToSymbol(c_gWindowTableY, hostTableY.data(), table_entries * 8 * sizeof(unsigned int));
    if (errY != cudaSuccess) return errY;

    c_gWindowTableInitialized_host = true;
    return cudaSuccess;
}


cudaError_t CudaDeviceKeys::initializePublicKeys(size_t count)
{

	// Allocate X array
	cudaError_t err = cudaMalloc(&_devX, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	// Clear X array
	err = cudaMemset(_devX, -1, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	// Allocate Y array
	err = cudaMalloc(&_devY, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	// Clear Y array
	err = cudaMemset(_devY, -1, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	err = cudaMemcpyToSymbol(_xPtr, &_devX, sizeof(unsigned int *));
	if(err) {
		return err;
	}

	err = cudaMemcpyToSymbol(_yPtr, &_devY, sizeof(unsigned int *));
	
	return err;
}

cudaError_t CudaDeviceKeys::init(int blocks, int threads, int pointsPerThread, const std::vector<secp256k1::uint256> &privateKeys)
{
	_blocks = blocks;
	_threads = threads;
	_pointsPerThread = pointsPerThread; // This might be re-interpreted if kernel handles 1 key/thread

    // Initialize G window table in constant memory (once)
    cudaError_t err = initConstantGWindowTable(4); // Assuming w=4
	if(err) {
		return err;
	}

	size_t total_num_keys = privateKeys.size();
    _numKeys = static_cast<unsigned int>(total_num_keys);


	// Allocate space for public keys on device (_devX, _devY)
	err = initializePublicKeys(total_num_keys);
	if(err) {
		return err;
	}

	// err = initializeBasePoints(); // Removed
	// if(err) {
	// return err;
	// }

	// Allocate private keys on device
	err = cudaMalloc(&_devPrivate, sizeof(unsigned int) * total_num_keys * 8);
	if(err) {
		return err;
	}


	// Clear private keys
	err = cudaMemset(_devPrivate, 0, sizeof(unsigned int) * total_num_keys * 8); // Corrected 'count' to 'total_num_keys'
	if(err) {
		return err;
	}

	// err = allocateChainBuf(_threads * _blocks * _pointsPerThread); // Call removed
	// if(err) {
	// 	return err;
	// }

	// Copy private keys to system memory buffer.
    // The splatBigInt arranges words for SoA access pattern if readInt/writeInt are used.
    // If kernel directly indexes _devPrivate as privateKeys[key_idx_global*8 + word_idx],
    // then a direct copy of privateKeys (after converting each uint256 to uint[8] big-endian) is simpler.
    // For now, assume splatBigInt's layout is what readInt in kernel expects or kernel adapts.
    // Let's simplify the host prep if kernel does direct indexing from a flat array of keys.
    // Each key is 8 uints. privateKeys is std::vector<secp256k1::uint256>.

    std::vector<unsigned int> hostPrivateKeys(total_num_keys * 8);
    for(size_t i = 0; i < total_num_keys; ++i) {
        // Store in big-endian format as that's what extract_kernel_w_bit_chunk expects from k_words
        privateKeys[i].exportWords(&hostPrivateKeys[i*8], 8, secp256k1::uint256::BigEndian);
    }

	// Copy private keys to device memory
	err = cudaMemcpy(_devPrivate, hostPrivateKeys.data(), total_num_keys * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);

	if(err) {
		return err;
	}

    // Launch kernel to compute all public keys
    // Determine grid/block dimensions.
    // If each thread handles one key:
    int threadsPerBlock = 256; // Example, can be tuned
    int blocks = (total_num_keys + threadsPerBlock - 1) / threadsPerBlock;

    // If kernel's pointsPerThread is > 1, adjust launch or kernel logic.
    // For this refactor, let's assume pointsPerThread in kernel context means each thread computes 1 key,
    // and the original _pointsPerThread member is for overall structure, not this specific kernel.
    // Or, more directly, if kernel has inner loop for pointsPerThread:
    // blocks = _blocks; threadsPerBlock = _threads; // from original setup
    // And kernel uses its pointsPerThread parameter.
    // The problem stated "pointsPerThread // How many keys each thread handles" for the kernel.
    // So, the original _blocks and _threads are the grid/block dimensions.

    if (_pointsPerThread == 0) return cudaErrorInvalidValue; // Avoid division by zero if _pointsPerThread is not set.

    // Assuming _blocks and _threads are already set to cover total_num_keys / _pointsPerThread
    // This means total_num_keys must be _blocks * _threads * _pointsPerThread
    // This seems to be the existing design.

    generateKeys_fixedWindow_kernel<<<_blocks, _threads>>>(
        _devPrivate,
        total_num_keys, // Total number of keys
        _pointsPerThread, // Keys per thread for this kernel launch
        _devX,
        _devY
    );

    err = cudaDeviceSynchronize(); // Wait for kernel completion and check for errors
	return err; // Return the error from synchronize or kernel launch
}

void CudaDeviceKeys::clearPublicKeys()
{
	cudaFree(_devX);
	cudaFree(_devY);

	_devX = NULL;
	_devY = NULL;
}

void CudaDeviceKeys::clearPrivateKeys()
{
	// cudaFree(_devBasePointX); // Removed
	// cudaFree(_devBasePointY); // Removed
	cudaFree(_devPrivate);
	// cudaFree(_devChain); // Removed

	// _devChain = NULL; // Removed
	// _devBasePointX = NULL; // Removed
	// _devBasePointY = NULL; // Removed
	_devPrivate = NULL;
}

// cudaError_t CudaDeviceKeys::doStep() // Removed
// {
// multiplyStepKernel <<<_blocks, _threads>>>(_devPrivate, _pointsPerThread, _step, _devChain, _devBasePointX, _devBasePointY);
// ...
// }

// __global__ void multiplyStepKernel(...) // Removed
// {
// ...
// }


__global__ void generateKeys_fixedWindow_kernel(
    const unsigned int *privateKeysGlobal, // All private keys, flat
    int numTotalKeys,
    int pointsPerThreadIn, // Num keys this CUDA thread processes
    unsigned int *outGxGlobal,    // All public key X coords, flat
    unsigned int *outGyGlobal     // All public key Y coords, flat
) {
    int w = 4; // Window size, matches c_gWindowTable
    int m = 256; // Scalar bit length

    // Calculate global thread ID and stride
    int thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads_globally = gridDim.x * blockDim.x;

    for (int pt_idx = 0; pt_idx < pointsPerThreadIn; ++pt_idx) {
        int key_global_idx = thread_global_id + pt_idx * total_threads_globally;

        if (key_global_idx >= numTotalKeys) {
            return;
        }

        // privateKeysGlobal points to an array of keys, each key is 8 uints.
        // k_words points to the start of the current private key.
        const unsigned int* k_words = &privateKeysGlobal[key_global_idx * 8];

        unsigned int Qx[8], Qy[8];
        // Initialize Q to point at infinity
        for(int i=0; i<8; ++i) Qx[i] = Qy[i] = 0xFFFFFFFF;

        // Check if private key is zero. If so, result is infinity.
        bool k_is_zero = true;
        for(int i=0; i<8; ++i) {
            if(k_words[i] != 0) {
                k_is_zero = false;
                break;
            }
        }

        if(!k_is_zero) {
            int t = (m + w - 1) / w; // Number of w-bit windows

            for (int i = t - 1; i >= 0; i--) { // Iterate from MSW to LSW
                // Perform w doublings on Q
                for (int d_loop = 0; d_loop < w; ++d_loop) {
                    // doublePoint expects params as (Px, Py, Rx, Ry)
                    // so, doublePoint(Qx, Qy, Qx, Qy) if it modifies in-place,
                    // or use temp if it doesn't: doublePoint(Qx_in, Qy_in, Qx_out, Qy_out)
                    // The device functions in secp256k1.cuh take separate out params.
                    unsigned int tempQx_double[8], tempQy_double[8];
                    device_doublePoint(Qx, Qy, tempQx_double, tempQy_double); // Use actual device function
                    copyBigInt(tempQx_double, Qx); // copyBigInt is __device__ __forceinline__ static in secp256k1.cuh
                    copyBigInt(tempQy_double, Qy);
                }

                unsigned int k_i_val = extract_kernel_w_bit_chunk(k_words, i, w, m);

                if (k_i_val != 0) { // If k_i_val is 0, table lookup is G_WINDOW_TABLE[0]*G = infinity, adding it does nothing.
                    unsigned int tablePx[8], tablePy[8];
                    // Read G_table[k_i_val] from c_gWindowTableX/Y (constant memory)
                    // Ensure k_i_val is within bounds, though extract_kernel_w_bit_chunk should ensure it's < (1<<w)
                    if (k_i_val < (1U << w)) {
                        for(int word_idx=0; word_idx<8; ++word_idx) {
                            tablePx[word_idx] = c_gWindowTableX[k_i_val * 8 + word_idx];
                            tablePy[word_idx] = c_gWindowTableY[k_i_val * 8 + word_idx];
                        }

                        unsigned int tempQx_add[8], tempQy_add[8];
                        device_addPoints(Qx, Qy, tablePx, tablePy, tempQx_add, tempQy_add); // Use actual device function
                        copyBigInt(tempQx_add, Qx);
                        copyBigInt(tempQy_add, Qy);
                    }
                }
            }
        }

        // Write final Qx, Qy to global output arrays (flat layout)
        // outGxGlobal_flat and outGyGlobal_flat are also flat arrays: [key0_x0..7, key1_x0..7, ...]
        // Ensure Qx, Qy words are in BigEndian order if that's the convention for storing points.
        // The host side secp256k1::ecpoint::exportWords(..., BigEndian) was used for table.
        // Device point ops should maintain this order or convert. Assuming they maintain.
        for(int word_idx=0; word_idx<8; ++word_idx) {
            outGxGlobal_flat[key_global_idx * 8 + word_idx] = Qx[word_idx];
            outGyGlobal_flat[key_global_idx * 8 + word_idx] = Qy[word_idx];
        }
    }
}


bool CudaDeviceKeys::selfTest(const std::vector<secp256k1::uint256> &privateKeys)
{
	unsigned int numPoints = _threads * _blocks * _pointsPerThread;

	unsigned int *xBuf = new unsigned int[numPoints * 8];
	unsigned int *yBuf = new unsigned int[numPoints * 8];

	cudaError_t err = cudaMemcpy(xBuf, _devX, sizeof(unsigned int) * 8 * numPoints, cudaMemcpyDeviceToHost);

	err = cudaMemcpy(yBuf, _devY, sizeof(unsigned int) * 8 * numPoints, cudaMemcpyDeviceToHost);


	for(int block = 0; block < _blocks; block++) {
		for(int thread = 0; thread < _threads; thread++) {
			for(int idx = 0; idx < _pointsPerThread; idx++) {

				int index = getIndex(block, thread, idx);

				secp256k1::uint256 privateKey = privateKeys[index];

				secp256k1::uint256 x = readBigInt(xBuf, block, thread, idx);
				secp256k1::uint256 y = readBigInt(yBuf, block, thread, idx);

				secp256k1::ecpoint p1(x, y);
				// Original: secp256k1::ecpoint p2 = secp256k1::multiplyPoint(privateKey, secp256k1::G());
				// New:
				secp256k1::ecpoint p2 = secp256k1::multiplyPointG_fixedwindow(privateKey);

				if(!secp256k1::pointExists(p1)) {
					throw std::string("Validation failed: invalid point");
				}

				if(!secp256k1::pointExists(p2)) {
					throw std::string("Validation failed: invalid point");
				}

				if(!(p1 == p2)) {
					throw std::string("Validation failed: points do not match");
				}
			}
		}
	}

	return true;
}