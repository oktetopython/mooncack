#ifndef _CUDA_KEY_SEARCH_DEVICE
#define _CUDA_KEY_SEARCH_DEVICE

#include "KeySearchDevice.h"
#include <vector>
#include <unordered_set> // Added for std::unordered_set
#include <cuda_runtime.h>
#include "secp256k1.h"
#include "CudaDeviceKeys.h"
#include "CudaHashLookup.h"
#include "CudaAtomicList.h"
#include "cudaUtil.h"

// Structures that exist on both host and device side
struct CudaDeviceResult {
    int thread;
    int block;
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
};

class CudaKeySearchDevice : public KeySearchDevice {

private:

    int _device;

    int _blocks;

    int _threads;

    int _pointsPerThread;

    int _compression;

    std::vector<KeySearchResult> _results;

    std::string _deviceName;

    cuda::CudaDeviceInfo _deviceInfo;

    secp256k1::uint256 _startExponent;

    uint64_t _iterations;

    void cudaCall(cudaError_t err);

    void generateStartingPoints();

    CudaDeviceKeys _deviceKeys;

    CudaAtomicList _resultList;

    CudaHashLookup _targetLookup;

    // void getResultsInternal(); // Removed

    // std::vector<hash160> _targets; // Old
    std::unordered_set<hash160, Hash160Hasher> _targets; // New

    bool isTargetInList(const unsigned int hash[5]);
    
    void removeTargetFromList(const unsigned int hash[5]);

    uint32_t getPrivateKeyOffset(int thread, int block, int point);

    secp256k1::uint256 _stride;

    bool verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5], bool compressed);

    // CUDA Streams
    cudaStream_t kernelStream;
    cudaStream_t memcpyStream;

    // CUDA Event for kernel completion
    cudaEvent_t kernelDoneEvent; // Renamed from kernelDoneEvent_d for consistency if this is the same event
                                 // Or if it's different, then kernelDoneEvent_d was correct.
                                 // The subtask said kernelDoneEvent_d, so I'll stick to that if it's distinct.
                                 // Re-reading: kernelDoneEvent is for kernel, memcpyDoneEvent_d for memcpy. So kernelDoneEvent is correct.

    // Double buffer and sync for async DtoH copy and processing
    std::vector<unsigned char> resultsProcessingBuffer_h; // Host buffer for CPU to process results
    std::vector<unsigned char> resultsDtoHBuffer_h;       // Host buffer for async DtoH copy
    cudaEvent_t memcpyDoneEvent_d;                      // Event to signal DtoH completion
    bool dtoHCopyInProgress_f;                          // Flag indicating a DtoH copy is outstanding
    int itemsInProcessingBuffer_i;                      // Number of items in resultsProcessingBuffer_h
    int itemsInDtoHBuffer_i;                            // Number of items copied into resultsDtoHBuffer_h

public:

    CudaKeySearchDevice(int device, int threads, int pointsPerThread, int blocks = 0);
    ~CudaKeySearchDevice(); // Destructor for stream cleanup

    virtual void init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride);

    virtual void doStep();

    virtual void setTargets(const std::set<KeySearchTarget> &targets);

    virtual size_t getResults(std::vector<KeySearchResult> &results);

    virtual uint64_t keysPerStep();

    virtual std::string getDeviceName();

    virtual void getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem);

    virtual secp256k1::uint256 getNextKey();
};

#endif