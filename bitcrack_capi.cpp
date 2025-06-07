#include "bitcrack_capi.h"

// C++ Standard Library includes
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstring> // For strncpy, etc.
#include <iostream> // For cerr in case of errors if Logger is not used directly

// BitCrack Core Headers
#include "secp256k1.h"
#include "CmdParse.h" // For PointCompressionType, though it's an enum, direct values might be okay.
#include "KeyFinder.h"
#include "DeviceManager.h"
#include "Logger.h" // Optional, for logging from C API layer
#include "AddressUtil.h" // For address validation if needed, though KeyFinder might do it.

// Assuming CUDA for now, generalize if OpenCL needs different handling here.
#ifdef BUILD_CUDA
#include "CudaKeySearchDevice.h"
#else
// Fallback or error if no GPU backend is built for the shared library
#error "BUILD_CUDA must be defined for the C API to use CudaKeySearchDevice"
// Or, provide a generic way to select device type if both can be in shared lib.
#endif


// Define the opaque struct
struct BitCrackSessionOpaque {
    KeyFinder* keyFinderInstance;
    KeySearchDevice* deviceInstance;
    DeviceManager::DeviceInfo deviceInfo; // Store selected device info

    secp256k1::uint256 startKey;
    secp256k1::uint256 endKey;
    int compressionMode;
    std::vector<std::string> targets; // Store targets

    // Threading and control
    std::thread searchThread;
    std::atomic<bool> stopSearchFlag;
    std::atomic<bool> searchRunningFlag;

    // C-style callback and user data
    PFN_BitCrackResultCallback resultCallback_c;
    void* resultCallbackUserData_c;
    std::mutex callback_mutex_c; // Renamed for clarity vs queue_mutex_c

    // Result queue for polling
    std::vector<BitCrackFoundKeyC> found_results_queue_c;
    std::mutex queue_mutex_c;

    // Configuration for KeySearchDevice
    int blocks;
    int threads;
    int pointsPerThread;
    bool device_configured;


    BitCrackSessionOpaque() :
        keyFinderInstance(nullptr),
        deviceInstance(nullptr),
        // deviceInfo will be set via bitcrack_set_device
        startKey(1),
        endKey(secp256k1::N - 1),
        compressionMode(PointCompressionType::COMPRESSED), // Default
        stopSearchFlag(false),
        searchRunningFlag(false),
        resultCallback_c(nullptr),
        resultCallbackUserData_c(nullptr),
        // callback_mutex_c is default constructed
        // queue_mutex_c is default constructed
        // found_results_queue_c is default constructed (empty)
        blocks(0), threads(0), pointsPerThread(0),
        device_configured(false)
    {}

    // Cleanup function to be called before destruction or re-initialization
    void cleanup_resources() {
        if (searchThread.joinable()) {
            stopSearchFlag = true; // Signal thread to stop
            // keyFinderInstance might need a stop() call if it's blocking internally
            if(keyFinderInstance) keyFinderInstance->stop();
            searchThread.join();
        }
        searchRunningFlag = false;

        delete keyFinderInstance;
        keyFinderInstance = nullptr;
        delete deviceInstance;
        deviceInstance = nullptr;
    }
};

// --- Helper: C++ Lambda to C Callback Bridge ---
// This lambda will be given to the C++ KeyFinder instance.
// It captures what's needed to call the C callback.
// (Defined within bitcrack_init_search or globally if preferred)


// --- Session Management ---
BITCRACK_API BitCrackSession bitcrack_create_session() {
    BitCrackSessionOpaque* session = new (std::nothrow) BitCrackSessionOpaque();
    if (!session) {
        // Optional: Log error if logger is available and configured
        // Logger::log(LogLevel::Error, "C API: Failed to allocate session memory");
        return nullptr;
    }
    return session;
}

BITCRACK_API void bitcrack_destroy_session(BitCrackSession session) {
    if (!session) return;
    session->cleanup_resources();
    delete session;
}

// --- Configuration ---
BITCRACK_API int bitcrack_set_device(BitCrackSession session, int device_id, int blocks_val, int threads_val, int points_val) {
    if (!session) return -1; // Invalid session

    // Validate device_id (optional here if init_search will do it, but good for early feedback)
    std::vector<DeviceManager::DeviceInfo> available_devices = DeviceManager::getDevices();
    bool found = false;
    for (const auto& dev : available_devices) {
        if (dev.id == device_id) {
            session->deviceInfo = dev;
            found = true;
            break;
        }
    }
    if (!found) {
        // Logger::log(LogLevel::Error, "C API: Invalid device_id specified: " + std::to_string(device_id));
        return -2; // Device not found
    }

    session->blocks = blocks_val;
    session->threads = threads_val;
    session->pointsPerThread = points_val;
    session->device_configured = true;

    // Logger::log(LogLevel::Info, "C API: Device set to " + session->deviceInfo.name);
    return 0; // Success
}

BITCRACK_API int bitcrack_set_keyspace(BitCrackSession session, const char* start_key_hex, const char* end_key_hex) {
    if (!session || !start_key_hex || !end_key_hex) return -1;
    try {
        session->startKey = secp256k1::uint256(start_key_hex);
        session->endKey = secp256k1::uint256(end_key_hex);
        if (session->startKey.isZero() || session->startKey.cmp(secp256k1::N) >= 0 ||
            session->endKey.isZero() || session->endKey.cmp(secp256k1::N) >= 0 ||
            session->startKey.cmp(session->endKey) > 0) {
            // Logger::log(LogLevel::Error, "C API: Invalid keyspace range.");
            return -2; // Invalid range
        }
    } catch (const std::string& err) { // Assuming uint256 constructor throws std::string on error
        // Logger::log(LogLevel::Error, "C API: Error parsing keyspace hex strings: " + err);
        return -3; // Parse error
    }  catch (const std::exception& e) {
        // Logger::log(LogLevel::Error, std::string("C API: Error parsing keyspace hex strings: ") + e.what());
        return -3; // Parse error
    }
    return 0; // Success
}

BITCRACK_API int bitcrack_add_targets(BitCrackSession session, const char* const* addresses, int num_addresses) {
    if (!session || !addresses || num_addresses <= 0) return -1;
    session->targets.clear();
    try {
        for (int i = 0; i < num_addresses; ++i) {
            if (addresses[i]) { // Basic null check
                // Optional: Validate address format here if desired, e.g. using Address::verifyAddress
                session->targets.push_back(std::string(addresses[i]));
            } else {
                 // Logger::log(LogLevel::Warning, "C API: Null address string provided in add_targets.");
            }
        }
    } catch (const std::exception& e) {
        // Logger::log(LogLevel::Error, std::string("C API: Exception while adding targets: ") + e.what());
        return -2; // Error adding targets
    }
    return 0; // Success
}

BITCRACK_API int bitcrack_set_compression_mode(BitCrackSession session, int mode) {
    if (!session) return -1;
    // Mode values directly correspond to PointCompressionType enum
    if (mode == PointCompressionType::COMPRESSED ||
        mode == PointCompressionType::UNCOMPRESSED ||
        mode == PointCompressionType::BOTH) {
        session->compressionMode = mode;
        return 0; // Success
    }
    // Logger::log(LogLevel::Error, "C API: Invalid compression mode: " + std::to_string(mode));
    return -2; // Invalid mode
}


// --- Internal C++ callback handler ---
void internal_cpp_result_callback(KeySearchResult cpp_result, BitCrackSessionOpaque* session) {
    if (session && session->resultCallback_c) {
        BitCrackFoundKeyC c_result;

        // Convert cpp_result to c_result
        strncpy(c_result.private_key_hex, cpp_result.privateKey.toString(16).c_str(), 64);
        c_result.private_key_hex[64] = '\0'; // Ensure null termination

        std::string pubKeyHex = cpp_result.publicKey.toString(cpp_result.compressed);
        strncpy(c_result.public_key_hex, pubKeyHex.c_str(), 130);
        c_result.public_key_hex[130] = '\0';

        strncpy(c_result.address_base58, cpp_result.address.c_str(), 59);
        c_result.address_base58[59] = '\0';

        c_result.is_compressed = cpp_result.compressed ? 1 : 0;

        // Lock mutex before calling C callback
        std::lock_guard<std::mutex> lock(session->callback_mutex_c);
        session->resultCallback_c(&c_result, session->resultCallbackUserData_c);
    }

    // Also, add to the internal queue for polling
    {
        std::lock_guard<std::mutex> queue_lock(session->queue_mutex_c);
        session->found_results_queue_c.push_back(c_result);
    }
}


BITCRACK_API int bitcrack_init_search(BitCrackSession session) {
    if (!session) return -1;
    if (!session->device_configured) {
        // Logger::log(LogLevel::Error, "C API: Device not configured before init_search.");
        return -2; // Device not set
    }
    if (session->targets.empty()) {
        // Logger::log(LogLevel::Error, "C API: No targets specified before init_search.");
        return -3; // No targets
    }

    // Cleanup any existing instances first (e.g. if re-initializing)
    session->cleanup_resources();
    session->stopSearchFlag = false; // Reset flag

    try {
#ifdef BUILD_CUDA
        // Use session->deviceInfo.id, session->blocks, etc.
        // If blocks/threads/points are 0, CudaKeySearchDevice constructor will use defaults or auto-tune.
        session->deviceInstance = new CudaKeySearchDevice(
            session->deviceInfo.id,
            session->threads,
            session->pointsPerThread,
            session->blocks
        );
#else
        // Logger::log(LogLevel::Error, "C API: No suitable KeySearchDevice implementation (CUDA not built).");
        return -4; // No backend
#endif

        if (!session->deviceInstance) {
             // Logger::log(LogLevel::Error, "C API: Failed to create KeySearchDevice.");
            return -5; // Device creation failed
        }

        // Stride is assumed to be 1 for now for C API, or needs a setter. Default is 1 in RunConfig.
        // Let's assume stride of 1 for now.
        secp256k1::uint256 stride(1);

        session->keyFinderInstance = new KeyFinder(
            session->startKey,
            session->endKey,
            session->compressionMode,
            session->deviceInstance,
            stride // Pass stride
        );

        if (session->resultCallback_c) {
            // Set up the C++ lambda to bridge to the C callback
            session->keyFinderInstance->setResultCallback(
                [session_ptr = session](KeySearchResult result) { // Capture session by pointer
                    internal_cpp_result_callback(result, session_ptr);
                }
            );
        }

        // Status callback could be set similarly if needed for C API
        // session->keyFinderInstance->setStatusCallback(...);
        // session->keyFinderInstance->setStatusInterval(...);

        // Set the stop flag for KeyFinder instance
        if(session->keyFinderInstance) {
            session->keyFinderInstance->setStopFlag(&(session->stopSearchFlag));
        }

        session->keyFinderInstance->init(); // This initializes the device, points, etc.
        session->keyFinderInstance->setTargets(session->targets);

    } catch (const KeySearchException& ex) {
        // Logger::log(LogLevel::Error, "C API: KeySearchException during init: " + ex.msg);
        session->cleanup_resources(); // Ensure cleanup on error
        return -6; // Init error
    } catch (const std::exception& ex) {
        // Logger::log(LogLevel::Error, std::string("C API: Std::exception during init: ") + ex.what());
        session->cleanup_resources();
        return -7; // Generic error
    }

    return 0; // Success
}

void key_finder_thread_wrapper(BitCrackSessionOpaque* session) {
    session->searchRunningFlag = true;
    try {
        if (session->keyFinderInstance) {
            // KeyFinder::run now uses the _stopFlagPtr member, which was set in init_search
            session->keyFinderInstance->run();
        }
    } catch (const KeySearchException& ex) {
        // Logger::log(LogLevel::Error, "C API: KeySearchException in search thread: " + ex.msg);
    } catch (const std::exception& ex) {
        // Logger::log(LogLevel::Error, std::string("C API: Std::exception in search thread: ") + ex.what());
    }
    session->searchRunningFlag = false;
}

BITCRACK_API int bitcrack_start_search_async(BitCrackSession session) {
    if (!session || !session->keyFinderInstance) return -1; // Not initialized or invalid session
    if (session->searchRunningFlag.load()) return -2; // Already running using .load() for atomic bool

    session->stopSearchFlag.store(false, std::memory_order_relaxed); // Reset stop flag before starting
    try {
        session->searchThread = std::thread(key_finder_thread_wrapper, session);
    } catch (const std::exception& e) {
        // Logger::log(LogLevel::Error, std::string("C API: Failed to launch search thread: ") + e.what());
        session->searchRunningFlag = false; // Ensure flag is correct
        return -3; // Thread launch failed
    }
    return 0; // Success
}

BITCRACK_API void bitcrack_stop_search(BitCrackSession session) {
    if (!session) return;
    session->stopSearchFlag.store(true, std::memory_order_relaxed); // Signal the thread by setting the atomic flag

    // The KeyFinder::stop() method also sets this flag now, and its internal _running flag.
    // Calling it here ensures its internal _running flag is also set, which might be checked earlier in its loop.
    if(session->keyFinderInstance) {
        session->keyFinderInstance->stop();
    }

    if (session->searchThread.joinable()) {
        session->searchThread.join(); // Wait for thread to finish
    }
    // searchRunningFlag is set to false by the thread_wrapper itself when it exits.
    // No need to set it here if join() was successful.
    // However, if join fails or thread was never started, ensuring it's false is okay.
    session->searchRunningFlag.store(false, std::memory_order_relaxed);
}

BITCRACK_API int bitcrack_is_search_running(BitCrackSession session) {
    if (!session) return 0;
    return session->searchRunningFlag.load() ? 1 : 0;
}

BITCRACK_API void bitcrack_set_result_callback(BitCrackSession session, PFN_BitCrackResultCallback callback, void* user_data) {
    if (!session) return;
    std::lock_guard<std::mutex> lock(session->callback_mutex_c); // Ensure thread safety when setting callback
    session->resultCallback_c = callback;
    session->resultCallbackUserData_c = user_data;
}

BITCRACK_API int bitcrack_poll_results(BitCrackSession session, BitCrackFoundKeyC* out_results, int max_results_to_fetch, int* num_results_fetched)
{
    if (!session || !out_results || max_results_to_fetch <= 0 || !num_results_fetched) {
        if(num_results_fetched) *num_results_fetched = 0;
        return -1; // Invalid arguments
    }

    std::lock_guard<std::mutex> lock(session->queue_mutex_c);

    int count_to_copy = 0;
    if (session->found_results_queue_c.empty()) {
        *num_results_fetched = 0;
        return 0; // Success, no results
    }

    count_to_copy = static_cast<int>(session->found_results_queue_c.size());
    if (count_to_copy > max_results_to_fetch) {
        count_to_copy = max_results_to_fetch;
    }

    for (int i = 0; i < count_to_copy; ++i) {
        out_results[i] = session->found_results_queue_c[i];
    }

    // Remove copied items from the front of the queue
    session->found_results_queue_c.erase(
        session->found_results_queue_c.begin(),
        session->found_results_queue_c.begin() + count_to_copy
    );

    *num_results_fetched = count_to_copy;
    return 0; // Success
}
