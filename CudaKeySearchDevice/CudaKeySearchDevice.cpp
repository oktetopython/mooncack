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

    // Block on kernel calls
    cudaCall(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    // Use a larger portion of shared memory for L1 cache
    cudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    generateStartingPoints();

    cudaCall(allocateChainBuf(_threads * _blocks * _pointsPerThread));

    // Set the incrementor
    secp256k1::ecpoint g = secp256k1::G();
    secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256((uint64_t)_threads * _blocks * _pointsPerThread) * _stride, g);

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
    uint64_t numKeys = (uint64_t)_blocks * _threads * _pointsPerThread;

    try {
        if(_iterations < 2 && _startExponent.cmp(numKeys) <= 0) {
            callKeyFinderKernel(_blocks, _threads, _pointsPerThread, true, _compression);
        } else {
            callKeyFinderKernel(_blocks, _threads, _pointsPerThread, false, _compression);
        }
    } catch(cuda::CudaException ex) {
        throw KeySearchException(ex.msg);
    }

    getResultsInternal();

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
    int count = _resultList.size();
    int actualCount = 0;
    if(count == 0) {
        return;
    }

    unsigned char *ptr = new unsigned char[count * sizeof(CudaDeviceResult)];

    _resultList.read(ptr, count);

    for(int i = 0; i < count; i++) {
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
    secp256k1::ecpoint g = secp256k1::G();

    secp256k1::ecpoint p = secp256k1::multiplyPoint(privateKey, g);

    if(!(p == publicKey)) {
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