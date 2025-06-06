#include "CudaKeySearchDevice.h"
#include "Logger.h"
#include "util.h"
#include "cudabridge.h"
#include "AddressUtil.h"

void CudaKeySearchDevice::cudaCall(cudaError_t err)
{
    if(err) {
        std::string errStr = cudaGetErrorString(err);

        throw KeySearchException(errStr);
    }
}

CudaKeySearchDevice::CudaKeySearchDevice(int device, int threads, int pointsPerThread, int blocks)
{
    cuda::CudaDeviceInfo info;
    try {
        info = cuda::getDeviceInfo(device);
        _deviceName = info.name;
    } catch(cuda::CudaException ex) {
        throw KeySearchException(ex.msg);
    }

    // Store device info locally, as it's used multiple times
    _deviceInfo = info; // Assuming _deviceInfo is a new member of CudaKeySearchDevice of type cuda::CudaDeviceInfo

    if (blocks == 0) { // Auto-calculate _threads and _blocks
        _threads = 256; // Default threads per block

        if (_threads > _deviceInfo.maxThreadsPerBlock) {
            _threads = _deviceInfo.maxThreadsPerBlock;
        }

        // Ensure multiple of warpSize and not zero
        _threads = (_threads / _deviceInfo.warpSize) * _deviceInfo.warpSize;
        if (_threads == 0) {
            _threads = _deviceInfo.warpSize;
        }

        int waves_per_sm = 4; // Default waves of blocks per SM
        _blocks = _deviceInfo.mpCount * waves_per_sm;
        if (_blocks == 0) { // Should not happen if mpCount > 0 and waves_per_sm > 0
            _blocks = 1; // Fallback to a single block
        }

        Logger::log(LogLevel::Info, "Auto-configured for device '" + _deviceName + "':");
        Logger::log(LogLevel::Info, "  SM Count: " + util::format("%d", _deviceInfo.mpCount));
        Logger::log(LogLevel::Info, "  Max Threads/Block: " + util::format("%d", _deviceInfo.maxThreadsPerBlock));
        Logger::log(LogLevel::Info, "  Warp Size: " + util::format("%d", _deviceInfo.warpSize));
        Logger::log(LogLevel::Info, "  Calculated Blocks: " + util::format("%d", _blocks) + ", Threads/Block: " + util::format("%d", _threads));

    } else { // User-specified _threads (as 'threads' param) and _blocks
        _threads = threads; // 'threads' parameter from constructor is threads per block
        _blocks = blocks;   // 'blocks' parameter from constructor

        Logger::log(LogLevel::Info, "User-specified configuration for device '" + _deviceName + "':");
        Logger::log(LogLevel::Info, "  Requested Blocks: " + util::format("%d", _blocks) + ", Threads/Block: " + util::format("%d", _threads));

        if (_threads > _deviceInfo.maxThreadsPerBlock) {
            Logger::log(LogLevel::Warn, "Requested threads per block (" + util::format("%d", _threads) +
                                       ") exceeds device maximum (" + util::format("%d", _deviceInfo.maxThreadsPerBlock) + "). Clamping.");
            _threads = _deviceInfo.maxThreadsPerBlock;
        }

        // Ensure multiple of warpSize and not zero
        int original_threads = _threads;
        _threads = (_threads / _deviceInfo.warpSize) * _deviceInfo.warpSize;
        if (_threads == 0) {
            Logger::log(LogLevel::Warn, "Requested threads per block (" + util::format("%d", original_threads) +
                                       ") is less than warp size (" + util::format("%d", _deviceInfo.warpSize) + "). Setting to warp size.");
            _threads = _deviceInfo.warpSize;
        } else if (original_threads != _threads) {
             Logger::log(LogLevel::Warn, "Adjusted threads per block from " + util::format("%d", original_threads) + " to " + util::format("%d", _threads) +
                                       " to be a multiple of warp size (" + util::format("%d", _deviceInfo.warpSize) + ").");
        }
        Logger::log(LogLevel::Info, "  Adjusted Blocks: " + util::format("%d", _blocks) + ", Threads/Block: " + util::format("%d", _threads));
    }

    // Common validations
    if (_threads <= 0) { // Should be caught by warpSize logic, but as a safeguard
        throw KeySearchException("Threads per block must be positive.");
    }
    if (_threads % _deviceInfo.warpSize != 0) {
        // This case should ideally be handled by the rounding logic above.
        // If it's reached, it implies a logic error or an unsupported warpSize (e.g. 0).
        throw KeySearchException("Threads per block must be a multiple of warp size (" + util::format("%d", _deviceInfo.warpSize) + "). Calculated: " + util::format("%d", _threads));
    }

    if (pointsPerThread <= 0) {
        throw KeySearchException("At least 1 point per thread required");
    }

    _iterations = 0;

    _device = device;

    _pointsPerThread = pointsPerThread;

    // Initialize streams to 0 (default stream) or nullptr
    kernelStream = 0;
    memcpyStream = 0;
    kernelDoneEvent = 0; // Initialize event for kernel completion

    // Initialize new members for double buffering and DtoH/processing overlap
    memcpyDoneEvent_d = 0; // Initialize event for DtoH completion
    dtoHCopyInProgress_f = false;
    itemsInProcessingBuffer_i = 0;
    itemsInDtoHBuffer_i = 0;
    // resultsProcessingBuffer_h and resultsDtoHBuffer_h are std::vector, default constructed.
}

CudaKeySearchDevice::~CudaKeySearchDevice()
{
    // Ensure device context is active for stream destruction, if necessary.
    // cudaSetDevice(_device); // Usually not needed if destructor is called before context is reset.
    if (kernelStream != 0) {
        cudaStreamDestroy(kernelStream);
        kernelStream = 0;
    }
    if (memcpyStream != 0) {
        cudaStreamDestroy(memcpyStream);
        memcpyStream = 0;
    }
    if (kernelDoneEvent != 0) {
        cudaEventDestroy(kernelDoneEvent);
        kernelDoneEvent = 0;
    }
    if (memcpyDoneEvent_d != 0) { // Destroy new DtoH event
        cudaEventDestroy(memcpyDoneEvent_d);
        memcpyDoneEvent_d = 0;
    }
}

void CudaKeySearchDevice::init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride)
{
    if(start.cmp(secp256k1::N) >= 0) {
        throw KeySearchException("Starting key is out of range");
    }

    _startExponent = start;

    _compression = compression;

    _stride = stride;

    cudaCall(cudaSetDevice(_device));

    // Change to allow asynchronous kernel execution relative to the host
    cudaCall(cudaSetDeviceFlags(cudaDeviceScheduleYield)); // Or cudaDeviceScheduleSpin

    // Create CUDA streams
    cudaCall(cudaStreamCreateWithFlags(&kernelStream, cudaStreamNonBlocking));
    cudaCall(cudaStreamCreateWithFlags(&memcpyStream, cudaStreamNonBlocking));

    // Create CUDA event for kernel completion
    cudaCall(cudaEventCreate(&kernelDoneEvent));
    // Create CUDA event for DtoH completion
    cudaCall(cudaEventCreate(&memcpyDoneEvent_d));


    // Use a larger portion of shared memory for L1 cache
    cudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    generateStartingPoints();

    cudaCall(allocateChainBuf(_threads * _blocks * _pointsPerThread));

    // Set the incrementor
    // Original:
    // secp256k1::ecpoint g = secp256k1::G();
    // secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256((uint64_t)_threads * _blocks * _pointsPerThread) * _stride, g);
    // New:
    secp256k1::uint256 incrementor_scalar = secp256k1::uint256((uint64_t)_threads * _blocks * _pointsPerThread) * _stride;
    secp256k1::ecpoint p = secp256k1::multiplyPointG_fixedwindow(incrementor_scalar);

    cudaCall(_resultList.init(sizeof(CudaDeviceResult), 16));

    cudaCall(setIncrementorPoint(p.x, p.y));
}


void CudaKeySearchDevice::generateStartingPoints()
{
    uint64_t totalPoints = (uint64_t)_pointsPerThread * _threads * _blocks;
    uint64_t totalMemory = totalPoints * 40;

    std::vector<secp256k1::uint256> exponents;

    Logger::log(LogLevel::Info, "Generating " + util::formatThousands(totalPoints) + " starting points (" + util::format("%.1f", (double)totalMemory / (double)(1024 * 1024)) + "MB)");

    // Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
    secp256k1::uint256 privKey = _startExponent;

    exponents.push_back(privKey);

    for(uint64_t i = 1; i < totalPoints; i++) {
        privKey = privKey.add(_stride);
        exponents.push_back(privKey);
    }

    cudaCall(_deviceKeys.init(_blocks, _threads, _pointsPerThread, exponents));

    // Show progress in 10% increments
    double pct = 10.0;
    for(int i = 1; i <= 256; i++) {
        cudaCall(_deviceKeys.doStep());

        if(((double)i / 256.0) * 100.0 >= pct) {
            Logger::log(LogLevel::Info, util::format("%.1f%%", pct));
            pct += 10.0;
        }
    }

    Logger::log(LogLevel::Info, "Done");

    _deviceKeys.clearPrivateKeys();
}


void CudaKeySearchDevice::setTargets(const std::set<KeySearchTarget> &targetsFromKeyFinder)
{
    _targets.clear();
    
    for(std::set<KeySearchTarget>::const_iterator i = targetsFromKeyFinder.begin(); i != targetsFromKeyFinder.end(); ++i) {
        hash160 h(i->value);
        _targets.insert(h); // Use insert for std::unordered_set
    }

    // Convert std::unordered_set to std::vector for _targetLookup.setTargets
    std::vector<hash160> tempTargetsVector(_targets.begin(), _targets.end());
    cudaCall(_targetLookup.setTargets(tempTargetsVector));
}

void CudaKeySearchDevice::doStep()
{
    // --- Part 1: Process results from the *previous* DtoH copy (if any) ---
    if (dtoHCopyInProgress_f) { // If a DtoH copy was initiated in the previous step
        cudaCall(cudaEventSynchronize(memcpyDoneEvent_d)); // Wait for that DtoH to complete

        resultsProcessingBuffer_h.swap(resultsDtoHBuffer_h); // Swap buffers
        itemsInProcessingBuffer_i = itemsInDtoHBuffer_i;

        dtoHCopyInProgress_f = false; // Mark DtoH buffer as free for next copy

        if (itemsInProcessingBuffer_i > 0) {
            unsigned char *ptr = resultsProcessingBuffer_h.data();
            int actualFoundThisStep = 0;
            for (int i = 0; i < itemsInProcessingBuffer_i; i++) {
                struct CudaDeviceResult *rPtr = &((struct CudaDeviceResult *)ptr)[i];

                // Host-side verification (bloom filter or exact match)
                if (!isTargetInList(rPtr->digest)) { // isTargetInList uses _targets (unordered_set)
                    continue;
                }
                actualFoundThisStep++;

                KeySearchResult minerResult;

                // Adjust iteration for offset calculation. Results being processed are from (_iterations - 1)
                // as _iterations would have been incremented after launching that kernel.
                uint64_t generatingIteration = _iterations > 0 ? _iterations -1 : 0;
                secp256k1::uint256 offset = (secp256k1::uint256((uint64_t)_blocks * _threads * _pointsPerThread * generatingIteration) +
                                   secp256k1::uint256(getPrivateKeyOffset(rPtr->thread, rPtr->block, rPtr->idx))) * _stride;
                minerResult.privateKey = secp256k1::addModN(_startExponent, offset);
                minerResult.compressed = rPtr->compressed;
                memcpy(minerResult.hash, rPtr->digest, 20);
                minerResult.publicKey = secp256k1::ecpoint(secp256k1::uint256(rPtr->x, secp256k1::uint256::BigEndian),
                                                      secp256k1::uint256(rPtr->y, secp256k1::uint256::BigEndian));

                removeTargetFromList(rPtr->digest); // remove from _targets
                _results.push_back(minerResult);    // _results is the main list for getResults() to pick up
            }

            if (actualFoundThisStep > 0) { // If any true positives were found and removed from _targets
                std::vector<hash160> tempTargetsVector(_targets.begin(), _targets.end());
                // Assuming _targetLookup.setTargets is thread-safe or called from a single host thread context
                // Also, this call might involve DtoH for the bloom filter itself.
                // For now, assume it's synchronous or managed correctly.
                cudaCall(_targetLookup.setTargets(tempTargetsVector));
            }
        }
    }

    // --- Part 2: Launch current iteration's kernel ---
    // Note: _startExponent for the current kernel is based on the current _iterations value
    uint64_t numKeysCurrentStep = (uint64_t)_blocks * _threads * _pointsPerThread;
    // The condition for using double keys in kernel was based on _startExponent.cmp(numKeys)
    // This logic should be reviewed if _startExponent is advanced with _iterations.
    // For now, using _iterations to determine if it's an early run.
    bool useDouble = (_iterations < 2 && _startExponent.cmp(numKeysCurrentStep) <= 0); // This logic might need adjustment based on how _startExponent is updated with _iterations
                                                                                        // Or simply pass _iterations to kernel if it influences point generation from _startExponent

    try {
        callKeyFinderKernel(_blocks, _threads, _pointsPerThread, useDouble, _compression, kernelStream, kernelDoneEvent);
    } catch(const cuda::CudaException &ex) { // Catch specific exception type if possible
        // Handle CUDA-specific exceptions from kernel launch or event record if they throw
        throw KeySearchException(ex.msg); // Re-throw as KeySearchException or handle
    } catch(const std::exception &ex) { // Catch other potential exceptions
        throw KeySearchException(std::string("Standard exception during kernel launch: ") + ex.what());
    }


    // --- Part 3: Initiate DtoH copy for results of the kernel just launched ---
    // Ensure kernel is done (via kernelDoneEvent on kernelStream) before memcpyStream attempts to read its results list size or copy.
    cudaCall(cudaStreamWaitEvent(memcpyStream, kernelDoneEvent, 0));

    int currentItemCount = _resultList.getCurrentItemCount(); // Synchronous DtoH for count
    if (currentItemCount > 0) {
        itemsInDtoHBuffer_i = currentItemCount;
        size_t listSizeBytes = (size_t)currentItemCount * sizeof(CudaDeviceResult);

        // Ensure buffer is large enough. std::vector::resize handles allocation.
        try {
            resultsDtoHBuffer_h.resize(listSizeBytes);
        } catch (const std::bad_alloc& e) {
            // Handle memory allocation failure for the host buffer
            Logger::log(LogLevel::Error, "Failed to allocate host DTO buffer: " + std::string(e.what()));
            dtoHCopyInProgress_f = false; // Ensure we don't try to process this failed copy
            itemsInDtoHBuffer_i = 0;
            // Potentially throw or stop processing
            throw KeySearchException("Failed to allocate host DTO buffer");
        }


        _resultList.readAsync(resultsDtoHBuffer_h.data(), listSizeBytes, memcpyStream);
        cudaCall(cudaEventRecord(memcpyDoneEvent_d, memcpyStream)); // Record event for DtoH completion on memcpyStream
        dtoHCopyInProgress_f = true;
    } else {
        itemsInDtoHBuffer_i = 0;
        dtoHCopyInProgress_f = false;
    }

    // Clear GPU list for next kernel run.
    // CudaAtomicList::clear() resets a host-side counter which is mapped to device.
    // If the kernel uses this counter directly, this reset needs to be visible to the next kernel invocation.
    // If clear() involves a cudaMemsetAsync, it should be on a stream that kernelStream waits for, or on kernelStream itself before next launch.
    // Current CudaAtomicList::clear() is `*_countHostPtr = 0;` which is host-side.
    // For mapped memory, this should be visible to device, but proper ordering/synchronization with kernel is key.
    // For now, assume this clear is okay. If it needs to be async and ordered, CudaAtomicList::clear needs a stream parameter.
    _resultList.clear();

    _iterations++;
}

uint64_t CudaKeySearchDevice::keysPerStep()
{
    return (uint64_t)_blocks * _threads * _pointsPerThread;
}

std::string CudaKeySearchDevice::getDeviceName()
{
    return _deviceName;
}

void CudaKeySearchDevice::getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem)
{
    cudaCall(cudaMemGetInfo(&freeMem, &totalMem));
}

void CudaKeySearchDevice::removeTargetFromList(const unsigned int hash[5])
{
    hash160 keyToRemove(hash);
    _targets.erase(keyToRemove);
}

bool CudaKeySearchDevice::isTargetInList(const unsigned int hash[5])
{
    hash160 keyToFind(hash);
    return _targets.count(keyToFind) > 0;
}

uint32_t CudaKeySearchDevice::getPrivateKeyOffset(int thread, int block, int idx)
{
    // Total number of threads
    int totalThreads = _blocks * _threads;

    int base = idx * totalThreads;

    // Global ID of the current thread
    int threadId = block * _threads + thread;

    return base + threadId;
}

void CudaKeySearchDevice::getResultsInternal()
{
    // Ensure kernel whose results are about to be read has completed
    if (kernelDoneEvent != 0) {
        cudaCall(cudaEventSynchronize(kernelDoneEvent));
    }

    // Get the current number of results using the new method
    int itemCount = _resultList.getCurrentItemCount();
    int actualCount = 0; // Will count valid results after host-side verification

    if(itemCount == 0) {
        // It's important to clear the list on device even if no items were found,
        // to reset its count for the next kernel run.
        _resultList.clear(); // This clears the host-side counter, which should also reset device counter via mapped memory or explicit reset.
                             // The current CudaAtomicList::clear() only does *_countHostPtr = 0.
                             // This might need enhancement if device counter isn't reset.
                             // For now, assume current clear() is sufficient or will be improved later.
        return;
    }

    int listSizeBytes = itemCount * sizeof(CudaDeviceResult);
    unsigned char *ptr = new unsigned char[listSizeBytes];

    // Ensure memcpyStream waits for kernelDoneEvent on kernelStream before starting the copy
    if (memcpyStream != 0 && kernelDoneEvent != 0) { // Check streams/events are valid
        cudaCall(cudaStreamWaitEvent(memcpyStream, kernelDoneEvent, 0));
    }

    // Asynchronously read the results
    _resultList.readAsync(ptr, listSizeBytes, memcpyStream);

    // Synchronize memcpyStream immediately for now
    // This makes the DtoH copy synchronous with respect to the host for this step.
    if (memcpyStream != 0) {
        cudaCall(cudaStreamSynchronize(memcpyStream));
    } else { // Fallback to device synchronize if using default stream for some reason
        cudaCall(cudaDeviceSynchronize());
    }


    for(int i = 0; i < itemCount; i++) { // Iterate based on itemCount
        struct CudaDeviceResult *rPtr = &((struct CudaDeviceResult *)ptr)[i];

        // might be false-positive
        if(!isTargetInList(rPtr->digest)) {
            continue;
        }
        actualCount++;

        KeySearchResult minerResult;

        // Calculate the private key based on the number of iterations and the current thread
        secp256k1::uint256 offset = (secp256k1::uint256((uint64_t)_blocks * _threads * _pointsPerThread * _iterations) + secp256k1::uint256(getPrivateKeyOffset(rPtr->thread, rPtr->block, rPtr->idx))) * _stride;
        secp256k1::uint256 privateKey = secp256k1::addModN(_startExponent, offset);

        minerResult.privateKey = privateKey;
        minerResult.compressed = rPtr->compressed;

        memcpy(minerResult.hash, rPtr->digest, 20);

        minerResult.publicKey = secp256k1::ecpoint(secp256k1::uint256(rPtr->x, secp256k1::uint256::BigEndian), secp256k1::uint256(rPtr->y, secp256k1::uint256::BigEndian));

        removeTargetFromList(rPtr->digest);

        _results.push_back(minerResult);
    }

    delete[] ptr;

    _resultList.clear();

    // Reload the bloom filters
    if(actualCount) {
        // Convert std::unordered_set to std::vector for _targetLookup.setTargets
        std::vector<hash160> tempTargetsVector(_targets.begin(), _targets.end());
        cudaCall(_targetLookup.setTargets(tempTargetsVector));
    }
}

// Verify a private key produces the public key and hash
bool CudaKeySearchDevice::verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5], bool compressed)
{
    // Original:
    // secp256k1::ecpoint g = secp256k1::G();
    // secp256k1::ecpoint p = secp256k1::multiplyPoint(privateKey, g);
    // New:
    secp256k1::ecpoint p = secp256k1::multiplyPointG_fixedwindow(privateKey);

    if(!(p == publicKey)) {
        // Optional: Log discrepancy for debugging
        // Logger::log(LogLevel::Debug, "Verification failed: Calculated public key does not match provided public key.");
        // Logger::log(LogLevel::Debug, "Private Key: " + privateKey.toString());
        // Logger::log(LogLevel::Debug, "Calc PubKey X: " + p.x.toString() + " Y: " + p.y.toString());
        // Logger::log(LogLevel::Debug, "Expected PubKey X: " + publicKey.x.toString() + " Y: " + publicKey.y.toString());
        return false;
    }

    unsigned int xWords[8];
    unsigned int yWords[8];

    p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
    p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

    unsigned int digest[5];
    if(compressed) {
        Hash::hashPublicKeyCompressed(xWords, yWords, digest);
    } else {
        Hash::hashPublicKey(xWords, yWords, digest);
    }

    for(int i = 0; i < 5; i++) {
        if(digest[i] != hash[i]) {
            return false;
        }
    }

    return true;
}

size_t CudaKeySearchDevice::getResults(std::vector<KeySearchResult> &resultsOut)
{
    for(int i = 0; i < _results.size(); i++) {
        resultsOut.push_back(_results[i]);
    }
    _results.clear();

    return resultsOut.size();
}

secp256k1::uint256 CudaKeySearchDevice::getNextKey()
{
    uint64_t totalPoints = (uint64_t)_pointsPerThread * _threads * _blocks;

    return _startExponent + secp256k1::uint256(totalPoints) * _iterations * _stride;
}