#include <stdio.h>
#include <fstream>
#include <iostream>

#include "KeyFinder.h"
#include "AddressUtil.h"
#include "util.h"
#include "secp256k1.h"
#include "CmdParse.h"
#include "Logger.h"
#include "ConfigFile.h"

#include "DeviceManager.h"

// spdlog for logging
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h" // For file logging

// Required for std::thread, std::vector, std::min, std::stringstream
// <vector> and <sstream> should be pulled in by previous changes or other includes.
// Add <thread> and <algorithm> explicitly if not covered.
#include <thread>       // For std::thread
#include <vector>       // For std::vector (though likely already included)
#include <algorithm>    // For std::min
#include <sstream>      // For std::stringstream (already added for device parsing)
#include <mutex>        // For std::mutex


#ifdef BUILD_CUDA
#include "CudaKeySearchDevice.h"
#endif

#ifdef BUILD_OPENCL
#include "CLKeySearchDevice.h"
#endif

typedef struct {
    // startKey is the first key. We store it so that if the --continue
    // option is used, the correct progress is displayed. startKey and
    // nextKey are only equal at the very beginning. nextKey gets saved
    // in the checkpoint file.
    secp256k1::uint256 startKey = 1;
    secp256k1::uint256 nextKey = 1;

    // The last key to be checked
    secp256k1::uint256 endKey = secp256k1::N - 1;

    uint64_t statusInterval = 1800;
    uint64_t checkpointInterval = 60000;

    unsigned int threads = 0;
    unsigned int blocks = 0;
    unsigned int pointsPerThread = 0;
    
    int compression = PointCompressionType::COMPRESSED;
 
    std::vector<std::string> targets;

    std::string targetsFile = "";

    std::string checkpointFile = "";

    // int device = 0; // Old: single device
    std::vector<int> devices; // New: vector of device IDs

    std::string resultsFile = "";

    uint64_t totalkeys = 0;
    unsigned int elapsed = 0;
    secp256k1::uint256 stride = 1;

    bool follow = false;
}RunConfig;

static RunConfig _config;

std::vector<DeviceManager::DeviceInfo> _devices;

// Global mutex for thread-safe console output and file access (simplified)
static std::mutex console_mutex;
// Note: For resultsFile, if multiple threads find keys simultaneously,
// util::appendToFile would need to be internally thread-safe or also protected by a mutex
// if direct file operations are not atomic / thread-safe by default on the OS.
// For this subtask, we assume appendToFile is sufficiently safe or conflicts are rare.

// Global logging configuration variables
static spdlog::level::level_enum g_log_level = spdlog::level::info; // Default log level
static std::string g_log_file_path = ""; // Default: no log file

// Global state for multi-GPU progress tracking and checkpointing
static std::map<int, secp256k1::uint256> g_device_progress_next_keys; // Stores nextKey for each physicalDeviceId
static std::map<int, secp256k1::uint256> g_checkpointed_device_progress; // Loaded from checkpoint file
static std::mutex g_device_progress_mutex; // Protects g_device_progress_next_keys and checkpoint file I/O

void writeCheckpoint(); // Updated signature

static uint64_t _lastUpdate = 0;
static uint64_t _runningTime = 0;
static uint64_t _startTime = 0;

/**
* Callback to display the private key
*/
void resultCallback(KeySearchResult info)
{
	if(_config.resultsFile.length() != 0) {
		Logger::log(LogLevel::Info, "Found key for address '" + info.address + "'. Written to '" + _config.resultsFile + "'");

		std::string s = info.address + " " + info.privateKey.toString(16) + " " + info.publicKey.toString(info.compressed);
		util::appendToFile(_config.resultsFile, s);

		return;
	}

	std::string logStr = "Address:     " + info.address + "\n";
	logStr += "Private key: " + info.privateKey.toString(16) + "\n";
	logStr += "Compressed:  ";

	if(info.compressed) {
		logStr += "yes\n";
	} else {
		logStr += "no\n";
	}

	logStr += "Public key:  \n";

	if(info.compressed) {
		logStr += info.publicKey.toString(true) + "\n";
	} else {
		logStr += info.publicKey.x.toString(16) + "\n";
		logStr += info.publicKey.y.toString(16) + "\n";
	}

	Logger::log(LogLevel::Info, logStr);
}

// Thread-safe version of resultCallback
void resultCallback_threadsafe(KeySearchResult info)
{
    // Use console_mutex for console output and general file writing (resultsFile)
    std::lock_guard<std::mutex> lock(console_mutex);

    if(_config.resultsFile.length() != 0) {
        // KeySearchResult doesn't have physicalDeviceId yet. Add it if needed, or use deviceName.
        // For now, using deviceName as physicalDeviceId might not be in KeySearchResult.
		spdlog::info("[{}] Found key for address '{}'. Written to '{}'", info.deviceName, info.address, _config.resultsFile);
		std::string s = info.address + " " + info.privateKey.toString(16) + " " + info.publicKey.toString(info.compressed);
		util::appendToFile(_config.resultsFile, s);
		return;
	}

	// Using deviceName as physicalDeviceId might not be in KeySearchResult.
    // Assuming KeySearchResult will be updated or deviceName is sufficient for context.
	std::string logStr = "[Dev " + info.deviceName + "] Address:     " + info.address + "\n";
	logStr += "[Dev " + info.deviceName + "] Private key: " + info.privateKey.toString(16) + "\n";
	logStr += "[Dev " + info.deviceName + "] Compressed:  ";

	if(info.compressed) {
		logStr += "yes\n";
	} else {
		logStr += "no\n";
	}
	logStr += "[Dev: " + info.deviceName + "] Public key:  \n";

	if(info.compressed) {
		logStr += info.publicKey.toString(true) + "\n";
	} else {
		logStr += info.publicKey.x.toString(16) + "\n";
		logStr += info.publicKey.y.toString(16) + "\n";
	}
	Logger::log(LogLevel::Info, logStr);
}


/**
Callback to display progress
*/
void statusCallback_threadsafe(KeySearchStatus info)
{
    // This function updates shared progress AND prints to console AND potentially writes checkpoints.
    // It needs to lock g_device_progress_mutex for progress map and checkpoint triggering.
    std::lock_guard<std::mutex> lock(g_device_progress_mutex);

    // Update this device's progress using physicalDeviceId (now available in KeySearchStatus)
    if(info.physicalDeviceId >= 0) {
        g_device_progress_next_keys[info.physicalDeviceId] = info.nextKey;
    } else {
        spdlog::warn("Status update received from device with uninitialized physicalDeviceId: {}", info.deviceName);
    }

	std::string speedStr;

	if(info.speed < 0.01) {
		speedStr = "< 0.01 MKey/s";
	} else {
		speedStr = util::format("%.2f", info.speed) + " MKey/s";
	}

	std::string totalStr = "(" + util::formatThousands(_config.totalkeys + info.total) + " total)";

	std::string timeStr = "[" + util::formatSeconds((unsigned int)((_config.elapsed + info.totalTime) / 1000)) + "]";

	std::string usedMemStr = util::format((info.deviceMemory - info.freeMemory) /(1024 * 1024));

	std::string totalMemStr = util::format(info.deviceMemory / (1024 * 1024));

    std::string targetStr = util::format(info.targets) + " target" + (info.targets > 1 ? "s" : "");


	// Fit device name in 10 characters for prefix, pad with spaces if less
	std::string devName = info.deviceName;
    if (devName.length() > 10) devName = devName.substr(0, 10);
    else devName += std::string(10 - devName.length(), ' ');

    const char *formatStr = NULL;

    // Always print newline for multi-device to prevent mangled output, or use a more sophisticated terminal manager
    // For simplicity, always newline. The "follow" concept might be harder with multiple lines.
    // formatStr = "[%s] %s/%sMB | %s %s %s %s\n";
    // Using a simpler format for multi-GPU to avoid complex cursor control for now.
    // Each device prints its own line.
    printf("[%s] Mem: %s/%sMB | Targets: %s | Speed: %s | Total: %s | Time: %s\n",
        devName.c_str(),
        usedMemStr.c_str(), totalMemStr.c_str(),
        targetStr.c_str(),
        speedStr.c_str(),
        totalStr.c_str(),
        timeStr.c_str());


    // Checkpoint logic:
    // This will be called by whichever thread's status update hits the interval.
    // It will use the global _config and _startTime/_lastUpdate.
    // This is not fully correct for multi-GPU checkpointing (e.g. nextKey for overall progress)
    // but is kept as per subtask instructions (minimal changes to checkpointing).
    // A proper multi-GPU checkpoint would need to aggregate progress or save per-device states.
    uint64_t t = util::getSystemTime();
    if(_config.checkpointFile.length() > 0 && (t - _lastUpdate >= _config.checkpointInterval)) {
        // This lock prevents multiple threads trying to write checkpoint simultaneously if their status updates align.
        // Only one thread should actually perform the write.
        // A simple way is to check _lastUpdate again inside the lock.
        // However, _lastUpdate itself is shared. This needs a more robust leader-election or dedicated checkpoint thread.
        // For now, this might lead to multiple checkpoint writes if interval is hit by several threads closely.
        // A quick fix: only one thread (e.g. for device 0 if it's active) handles checkpointing.
        // This is still not perfect.
        // Let's assume for now the existing global _lastUpdate offers some protection against too frequent writes.
        // The problem is if info.nextKey is from a GPU that is further ahead than others.
        // Current problem statement says: "writeCheckpoint still uses global _config".

        // Simplification: let any thread trigger it, but it saves based on global _config.
        // This part of the subtask is "Conceptual for now" / "minimal changes".
        // The actual nextKey to save should be the minimum of all nextKeys from active threads.
        // The actual nextKey to save for overall progress should be the minimum of all g_device_progress_next_keys
        // for active devices. writeCheckpoint() will handle this logic.
        spdlog::info("[Dev {} ({})] Checkpoint triggered by this device's status update.", info.physicalDeviceId, info.deviceName);
        writeCheckpoint(); // Call new checkpoint function (no specific key passed)
        _lastUpdate = t; // This update is global
    }
    fflush(stdout); // Ensure printf is flushed
}

/**
 * Parses the start:end key pair. Possible values are:
 start
 start:end
 start:+offset
 :end
 :+offset
 */
bool parseKeyspace(const std::string &s, secp256k1::uint256 &start, secp256k1::uint256 &end)
{
    size_t pos = s.find(':');

    if(pos == std::string::npos) {
        start = secp256k1::uint256(s);
        end = secp256k1::N - 1;
    } else {
        std::string left = s.substr(0, pos);

        if(left.length() == 0) {
            start = secp256k1::uint256(1);
        } else {
            start = secp256k1::uint256(left);
        }

        std::string right = s.substr(pos + 1);

        if(right[0] == '+') {
            end = start + secp256k1::uint256(right.substr(1));
        } else {
            end = secp256k1::uint256(right);
        }
    }

    return true;
}

void usage()
{
    printf("BitCrack OPTIONS [TARGETS]\n");
    printf("Where TARGETS is one or more addresses\n\n");
	
    printf("--help                  Display this message\n");
    printf("-c, --compressed        Use compressed points\n");
    printf("-u, --uncompressed      Use Uncompressed points\n");
    printf("--compression  MODE     Specify compression where MODE is\n");
    printf("                          COMPRESSED or UNCOMPRESSED or BOTH\n");
    printf("-d, --device ID         Use device ID\n");
    printf("-b, --blocks N          N blocks\n");
    printf("-t, --threads N         N threads per block\n");
    printf("-p, --points N          N points per thread\n");
    printf("-i, --in FILE           Read addresses from FILE, one per line\n");
    printf("-o, --out FILE          Write keys to FILE\n");
    printf("-f, --follow            Follow text output\n");
    printf("--list-devices          List available devices\n");
    printf("--keyspace KEYSPACE     Specify the keyspace:\n");
    printf("                          START:END\n");
    printf("                          START:+COUNT\n");
    printf("                          START\n");
    printf("                          :END\n"); 
    printf("                          :+COUNT\n");
    printf("                        Where START, END, COUNT are in hex format\n");
    printf("--stride N              Increment by N keys at a time\n");
    printf("--share M/N             Divide the keyspace into N equal shares, process the Mth share\n");
    printf("--continue FILE         Save/load progress from FILE\n");
    printf("--checkpoint-interval S Interval in SECONDS to write checkpoint file (default 60)\n");
    printf("--logfile FILE          Redirect log output to FILE\n");
    printf("--loglevel LEVEL        Set log level (trace, debug, info, warn, error, critical, off)\n");
}


/**
 Finds default parameters depending on the device
 */
typedef struct {
	int threads;
	int blocks;
	int pointsPerThread;
}DeviceParameters;

DeviceParameters getDefaultParameters(const DeviceManager::DeviceInfo &device)
{
	DeviceParameters p;
	p.threads = 256;
    p.blocks = 32;
	p.pointsPerThread = 32;

	return p;
}

static KeySearchDevice *getDeviceContext(DeviceManager::DeviceInfo &deviceInfo, int blocks, int threads, int pointsPerThread)
{
#ifdef BUILD_CUDA
    if(deviceInfo.type == DeviceManager::DeviceType::CUDA) {
        return new CudaKeySearchDevice(deviceInfo.id, threads, pointsPerThread, blocks);
    }
#endif

#ifdef BUILD_OPENCL
    if(deviceInfo.type == DeviceManager::DeviceType::OpenCL) {
        return new CLKeySearchDevice(deviceInfo.id, threads, pointsPerThread, blocks);
    }
#endif

    return NULL;
}

// Placeholder for the thread function
static void device_thread_function(DeviceManager::DeviceInfo devInfo,
                                   secp256k1::uint256 startKey,
                                   secp256k1::uint256 endKey,
                                   RunConfig thread_config,
                                   const std::vector<std::string>& targets_list,
                                   const std::string& targets_file_path)
{
    std::string devPrefix = "[Dev " + std::to_string(devInfo.id) + "] ";
    spdlog::info("{}{} Processing keys from {} to {}",
                devPrefix, devInfo.name, startKey.toString(16), endKey.toString(16));

    // Initialize this device's progress in the global map with its actual starting key
    {
        std::lock_guard<std::mutex> lock(g_device_progress_mutex);
        // Only set if not already present from a checkpoint, or if checkpointed value is before our actual start.
        // This ensures that if a checkpoint dictates a later start, we honor it.
        // However, run() function is now responsible for adjusting startKey from checkpoint.
        // So, here, we just record what this thread *thinks* its starting key is.
        // writeCheckpoint will then save the *latest* g_device_progress_next_keys.
        g_device_progress_next_keys[devInfo.id] = startKey;
    }

    try {
        // Determine device parameters (use thread_config's values if set, otherwise use defaults for this device)
        DeviceParameters params = getDefaultParameters(devInfo);
        unsigned int actual_blocks = (thread_config.blocks == 0) ? params.blocks : thread_config.blocks;
        unsigned int actual_threads = (thread_config.threads == 0) ? params.threads : thread_config.threads;
        unsigned int actual_points = (thread_config.pointsPerThread == 0) ? params.pointsPerThread : thread_config.pointsPerThread;

        // Logger::log(LogLevel::Info, devPrefix + "Using B: " + std::to_string(actual_blocks) +
        //            ", T: " + std::to_string(actual_threads) +
        //            ", P: " + std::to_string(actual_points));
        spdlog::info("{}Using B: {}, T: {}, P: {}", devPrefix, actual_blocks, actual_threads, actual_points);


        KeySearchDevice *d = getDeviceContext(devInfo, actual_blocks, actual_threads, actual_points);
        if (!d) {
            // Logger::log(LogLevel::Error, devPrefix + "Failed to get device context.");
            spdlog::error("{}Failed to get device context.", devPrefix);
            return;
        }

        KeyFinder f(startKey, endKey, thread_config.compression, d, thread_config.stride, devInfo.id); // Pass devInfo.id as physicalDeviceId

        f.setResultCallback(resultCallback_threadsafe);
        f.setStatusInterval(thread_config.statusInterval);
        f.setStatusCallback(statusCallback_threadsafe);

        f.init();

        if(!targets_file_path.empty()) {
            f.setTargets(targets_file_path);
        } else {
            f.setTargets(targets_list);
        }

        // Logger::log(LogLevel::Info, devPrefix + "Starting search...");
        spdlog::info("{}Starting search...", devPrefix);
        f.run();
        // Logger::log(LogLevel::Info, devPrefix + "Search finished.");
        spdlog::info("{}Search finished.", devPrefix);

        delete d;
    } catch(const KeySearchException &ex) {
        // Logger::log(LogLevel::Error, devPrefix + "Error: " + ex.msg);
        spdlog::error("{}KeySearchException: {}", devPrefix, ex.msg);
    } catch(const std::exception &exStd) {
         // Logger::log(LogLevel::Error, devPrefix + "Standard Exception: " + exStd.what());
         spdlog::error("{}Std::exception: {}", devPrefix, exStd.what());
    }
}


static void printDeviceList(const std::vector<DeviceManager::DeviceInfo> &devices)
{
    for(int i = 0; i < devices.size(); i++) {
        printf("ID:     %d\n", devices[i].id);
        printf("Name:   %s\n", devices[i].name.c_str());
        printf("Memory: %lldMB\n", devices[i].memory / ((uint64_t)1024 * 1024));
        printf("Compute units: %d\n", devices[i].computeUnits);
        printf("\n");
    }
}

bool readAddressesFromFile(const std::string &fileName, std::vector<std::string> &lines)
{
    if(fileName == "-") {
        return util::readLinesFromStream(std::cin, lines);
    } else {
        return util::readLinesFromStream(fileName, lines);
    }
}

int parseCompressionString(const std::string &s)
{
    std::string comp = util::toLower(s);

    if(comp == "both") {
        return PointCompressionType::BOTH;
    }

    if(comp == "compressed") {
        return PointCompressionType::COMPRESSED;
    }

    if(comp == "uncompressed") {
        return PointCompressionType::UNCOMPRESSED;
    }

    throw std::string("Invalid compression format: '" + s + "'");
}

static std::string getCompressionString(int mode)
{
    switch(mode) {
    case PointCompressionType::BOTH:
        return "both";
    case PointCompressionType::UNCOMPRESSED:
        return "uncompressed";
    case PointCompressionType::COMPRESSED:
        return "compressed";
    }

    throw std::string("Invalid compression setting '" + util::format(mode) + "'");
}

// --- Logging Setup ---
void SetupGlobalLogging() {
    try {
        // Create a color console logger.
        // This will be the default logger.
        auto console_logger = spdlog::stdout_color_mt("console");
        spdlog::set_default_logger(console_logger);
        // Default console logger setup is done here if no file logging,
        // or as part of multi-sink logger if file logging is enabled.
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");
        console_sink->set_level(g_log_level); // Apply global level or specific for console

        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(console_sink);

        if (!g_log_file_path.empty()) {
            try {
                auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(g_log_file_path, true); // true = truncate file
                file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [thread %t] %v"); // Example: no color for file
                file_sink->set_level(g_log_level); // Apply global level or specific for file
                sinks.push_back(file_sink);
                spdlog::info("Logging to file: {}", g_log_file_path);
            } catch (const spdlog::spdlog_ex& ex) {
                // Log to console (if available) that file sink failed
                spdlog::get("console")->error("Log file setup failed: {}", ex.what());
                // Or std::cerr if console logger itself might have failed earlier (though unlikely if we reach here)
                // std::cerr << "Log file setup failed: " << ex.what() << std::endl;
            }
        }

        auto combined_logger = std::make_shared<spdlog::logger>("multi_sink", begin(sinks), end(sinks));
        combined_logger->set_level(g_log_level); // Master level for the logger itself
        combined_logger->flush_on(g_log_level);  // Flush based on the global level for this combined logger
                                                 // Or choose a specific level, e.g., spdlog::level::warn or spdlog::level::err

        spdlog::set_default_logger(combined_logger);

    } catch (const spdlog::spdlog_ex& ex) {
        // Fallback to std::cerr if primary spdlog setup (e.g. console_sink) fails
        std::cerr << "Root log initialization failed: " << ex.what() << std::endl;
    }
}


// void writeCheckpoint(secp256k1::uint256 nextKey) // Old signature
void writeCheckpoint() // New signature
{
    std::lock_guard<std::mutex> lock(g_device_progress_mutex); // Protect map access and file I/O

    if(_config.checkpointFile.empty()) {
        spdlog::debug("Checkpoint file path is empty, skipping writeCheckpoint.");
        return;
    }

    std::ofstream tmp(_config.checkpointFile, std::ios::out);
    if(!tmp.is_open()) {
        spdlog::error("Failed to open checkpoint file for writing: {}", _config.checkpointFile);
        return;
    }

    tmp << "start=" << _config.startKey.toString(16) << std::endl;

    secp256k1::uint256 overall_next_key = _config.endKey;
    bool progress_recorded_for_any_configured_device = false;

    // Iterate through devices configured for *this run* to determine overall_next_key
    if (!_config.devices.empty()) {
        bool first_active_device_found = false;
        for (int dev_id : _config.devices) {
            auto it = g_device_progress_next_keys.find(dev_id);
            if (it != g_device_progress_next_keys.end()) {
                 if (!first_active_device_found || it->second.cmp(overall_next_key) < 0) {
                    overall_next_key = it->second;
                    first_active_device_found = true;
                }
                progress_recorded_for_any_configured_device = true;
            }
        }
        // If no configured & active device reported progress yet, fallback to global _config.nextKey
        if (!progress_recorded_for_any_configured_device) {
             overall_next_key = _config.nextKey;
        }
    } else { // No devices configured for this run, use global _config.nextKey (should not happen if run() checks)
        overall_next_key = _config.nextKey;
    }

    tmp << "next=" << overall_next_key.toString(16) << std::endl;
    tmp << "end=" << _config.endKey.toString(16) << std::endl;
    tmp << "blocks=" << _config.blocks << std::endl;
    tmp << "threads=" << _config.threads << std::endl;
    tmp << "points=" << _config.pointsPerThread << std::endl;
    tmp << "compression=" << getCompressionString(_config.compression) << std::endl;
    tmp << "elapsed=" << (_config.elapsed + (util::getSystemTime() - _startTime)) / 1000 << std::endl;
    tmp << "stride=" << _config.stride.toString() << std::endl;

    // Save per-device progress for devices configured in *this run*
    tmp << "num_devices=" << _config.devices.size() << std::endl;
    for(size_t i = 0; i < _config.devices.size(); ++i) {
        int physical_id = _config.devices[i];
        secp256k1::uint256 next_key_for_device;
        auto it = g_device_progress_next_keys.find(physical_id);
        if(it != g_device_progress_next_keys.end()) {
            next_key_for_device = it->second;
        } else {
            // This device is in _config.devices but hasn't reported progress or its entry wasn't pre-initialized.
            // This implies it should (re)start from its original calculated segment start.
            // Saving 0 here is a signal to readCheckpointFile/run to recalculate its start.
            next_key_for_device = secp256k1::uint256(0);
            spdlog::warn("Device {} is configured for run, but no progress entry in g_device_progress_next_keys during checkpoint. Its segment will start from original calculation or global 'next'.", physical_id);
        }
        tmp << "device_" << i << "_id=" << physical_id << std::endl;
        tmp << "device_" << i << "_next_key=" << next_key_for_device.toString(16) << std::endl;
    }

    tmp.close();
    spdlog::info("Checkpoint saved to '{}'. Overall 'next' key (approx min of active devices): {}.", _config.checkpointFile, overall_next_key.toString(16));
}

void readCheckpointFile()
{
    if(_config.checkpointFile.empty()) { // Use .empty() for std::string
        return;
    }

    ConfigFileReader reader(_config.checkpointFile);

    if(!reader.exists()) {
        spdlog::info("Checkpoint file '{}' not found, starting fresh.", _config.checkpointFile);
        return;
    }

    spdlog::info("Loading checkpoint from '{}'", _config.checkpointFile);
    g_checkpointed_device_progress.clear(); // Clear any previous/stale data

    try {
        std::map<std::string, ConfigFileEntry> entries = reader.read();

        // Helper lambda to get value or use default if key is optional and not found
        auto get_val = [&](const std::string& key, bool optional = false, const std::string& default_val = "") -> std::string {
            auto it = entries.find(key);
            if (it == entries.end()) {
                if(optional) return default_val;
                throw std::runtime_error("Checkpoint key missing: " + key);
            }
            return it->second.value;
        };

        _config.startKey = secp256k1::uint256(get_val("start"));
        // This 'next' is the overall progress point. It will be used if per-device info isn't available for a device.
        _config.nextKey = secp256k1::uint256(get_val("next"));
        _config.endKey = secp256k1::uint256(get_val("end"));

        // Optional fields, use current _config values as default if not in checkpoint or if parsing fails.
        // This allows user to override checkpointed B/T/P with command line args.
        if(_config.threads == 0) _config.threads = util::parseUInt32(get_val("threads", true, "0"));
        if(_config.blocks == 0) _config.blocks = util::parseUInt32(get_val("blocks", true, "0"));
        if(_config.pointsPerThread == 0) _config.pointsPerThread = util::parseUInt32(get_val("points", true, "0"));

        _config.compression = parseCompressionString(get_val("compression", true, getCompressionString(_config.compression)));
        _config.elapsed = util::parseUInt64(get_val("elapsed", true, "0")) * 1000; // Convert stored seconds back to ms
        _config.stride = secp256k1::uint256(get_val("stride", true, _config.stride.toString()));


        std::string num_devices_str = get_val("num_devices", true, "0");
        int num_devices_chk = 0;
        if (!num_devices_str.empty() && num_devices_str != "0") { // Ensure not empty before stoi
            num_devices_chk = std::stoi(num_devices_str);
        }

        if(num_devices_chk > 0) {
            spdlog::debug("Reading progress for {} devices from checkpoint.", num_devices_chk);
            for(int i = 0; i < num_devices_chk; ++i) {
                std::string dev_id_key = "device_" + std::to_string(i) + "_id";
                std::string dev_next_key_str = "device_" + std::to_string(i) + "_next_key";

                int physical_id = std::stoi(get_val(dev_id_key));
                secp256k1::uint256 next_key(get_val(dev_next_key_str));

                if(next_key.isZero()){
                    spdlog::warn("Device {} had a zero next_key in checkpoint. It will start from its calculated segment start or overall 'nextKey'.", physical_id);
                } else {
                    g_checkpointed_device_progress[physical_id] = next_key;
                }
            }
        } else if (entries.count("device")) { // Handle legacy checkpoint
            int legacy_device_id = std::stoi(get_val("device"));
            g_checkpointed_device_progress[legacy_device_id] = _config.nextKey; // Use overall 'next' for this device
            spdlog::info("Legacy checkpoint format detected. Device {} will attempt to resume from overall nextKey {}.", legacy_device_id, _config.nextKey.toString(16));
        }

        // _config.totalkeys will be recalculated based on effective start key in run()
        _config.totalkeys = 0; // Reset, as this old value is based on global nextKey.

        spdlog::info("Checkpoint loaded. Overall 'nextKey' (used for initial division or fallback): {}.", _config.nextKey.toString(16));
        if (!g_checkpointed_device_progress.empty()) {
            std::string per_device_log_msg = "Loaded per-device progress: ";
            for(const auto& pair : g_checkpointed_device_progress) {
                per_device_log_msg += "Dev " + std::to_string(pair.first) + ": " + pair.second.toString(16) + "; ";
            }
            spdlog::info(per_device_log_msg);
        }

    } catch (const std::exception& e) {
        spdlog::error("Error reading checkpoint file '{}': {}. Progress may not be fully restored. Command-line keyspace will be used.", _config.checkpointFile, e.what());
        g_checkpointed_device_progress.clear(); // Clear partial progress on error
        // Do not override _config.nextKey, startKey, endKey if checkpoint is corrupt for these.
        // Let command-line or default keyspace be the fallback by not re-throwing.
    }
}

int run()
{
    // Default to device 0 if no devices were specified by user
    if (_config.devices.empty()) {
        if (!_devices.empty() && _devices[0].id >=0) {
            spdlog::info("No device specified via -d, defaulting to device {} ({})", _devices[0].id, _devices[0].name);
            _config.devices.push_back(_devices[0].id);
        } else {
            spdlog::error("No CUDA/OpenCL devices detected or available to default to.");
            return 1;
        }
    }

    std::string dev_list_str_log; // Renamed to avoid conflict if dev_list_str is used later
    for(size_t i=0; i < _config.devices.size(); ++i) {
        dev_list_str_log += std::to_string(_config.devices[i]);
        if(i < _config.devices.size() - 1) dev_list_str_log += ",";
    }
    spdlog::info("Using {} GPU(s): {}", _config.devices.size(), dev_list_str_log);

    spdlog::info("Compression: {}", getCompressionString(_config.compression));
    // _config.nextKey is the overall starting point, potentially from a checkpoint's global 'next'
    spdlog::info("Overall Configured Keyspace Start: {}", _config.startKey.toString(16)); // The absolute start of the range
    spdlog::info("Effective Run Start Key (from checkpoint or config): {}", _config.nextKey.toString(16));
    spdlog::info("Overall Configured Keyspace End:   {}", _config.endKey.toString(16));
    spdlog::info("Stride: {}", _config.stride.toString());

    _startTime = util::getSystemTime(); // _config.elapsed should be loaded from checkpoint
    _lastUpdate = _startTime;

    std::vector<std::thread> worker_threads;

    // The effective starting key for this run, considering checkpoint.
    secp256k1::uint256 current_run_overall_start_key = _config.nextKey;

    if (current_run_overall_start_key.cmp(_config.endKey) > 0 &&
        !(_config.startKey == _config.endKey && current_run_overall_start_key == _config.startKey && current_run_overall_start_key ==1 )) { // Special case for 1 key scan
        spdlog::error("Effective start key {} is greater than end key {}. Nothing to scan.", current_run_overall_start_key.toString(16), _config.endKey.toString(16));
        return 1;
    }
    secp256k1::uint256 total_keys_to_scan_this_run = (_config.endKey - current_run_overall_start_key) + 1;

    unsigned int num_gpus = _config.devices.size();

    if (num_gpus == 0) {
        spdlog::error("Configuration error: No GPUs selected for processing (num_gpus is 0).");
        return 1;
    }
    if (total_keys_to_scan_this_run.isZero() && !(_config.startKey == _config.endKey && current_run_overall_start_key == _config.startKey && current_run_overall_start_key ==1)) {
         spdlog::info("Total keys to scan for this run is zero. Nothing to do.");
         return 0;
    }
     if (total_keys_to_scan_this_run.isZero() && (_config.startKey == _config.endKey && current_run_overall_start_key == _config.startKey && current_run_overall_start_key ==1) ){
        total_keys_to_scan_this_run = 1; // Scan the single key "1"
        spdlog::info("Scanning single key '1'.");
    }

    secp256k1::uint256 keys_per_gpu_base = total_keys_to_scan_this_run / num_gpus;
    secp256k1::uint256 remainder_keys_uint256 = total_keys_to_scan_this_run % num_gpus;
    uint64_t remainder_keys = remainder_keys_uint256.toUint64Safe(); // Use safe conversion

    secp256k1::uint256 next_segment_start_key = current_run_overall_start_key;

    for (unsigned int i = 0; i < num_gpus; ++i) {
        int actual_device_id = _config.devices[i];
        DeviceManager::DeviceInfo deviceInfo;
        // Find DeviceInfo for actual_device_id
        auto it_dev = std::find_if(_devices.begin(), _devices.end(),
                                   [actual_device_id](const DeviceManager::DeviceInfo& d){ return d.id == actual_device_id; });
        if(it_dev == _devices.end()){
            spdlog::error("Internal Error: Device ID {} not found during thread setup. Should have been caught in main().", actual_device_id);
            continue;
        }
        deviceInfo = *it_dev;

        // Calculate this device's theoretical segment of the *remaining* keyspace for this run
        secp256k1::uint256 num_keys_for_this_gpu = keys_per_gpu_base;
        if (i < remainder_keys) {
            num_keys_for_this_gpu = num_keys_for_this_gpu + 1;
        }

        if (num_keys_for_this_gpu.isZero()) {
            spdlog::info("[Dev {}] No keys assigned to this GPU in its share, skipping.", deviceInfo.id);
            continue;
        }

        secp256k1::uint256 segment_start_key = next_segment_start_key;
        secp256k1::uint256 segment_end_key = segment_start_key + num_keys_for_this_gpu - 1;

        // Adjust if segment_end_key exceeds overall run end key
        if(segment_end_key.cmp(_config.endKey) > 0) {
            segment_end_key = _config.endKey;
        }

        // Effective start key for this thread, considering checkpointed progress for this device
        secp256k1::uint256 thread_actual_start_key = segment_start_key;
        auto checkpoint_it = g_checkpointed_device_progress.find(actual_device_id);
        if (checkpoint_it != g_checkpointed_device_progress.end()) {
            const secp256k1::uint256& checkpointed_next = checkpoint_it->second;
            // Ensure checkpointed_next is within this device's calculated segment for this run
            if (!checkpointed_next.isZero() &&
                checkpointed_next.cmp(segment_start_key) >= 0 &&
                checkpointed_next.cmp(segment_end_key) <= 0) {
                thread_actual_start_key = checkpointed_next;
                spdlog::info("[Dev {}] Resuming from checkpoint: {}", actual_device_id, thread_actual_start_key.toString(16));
            } else if (!checkpointed_next.isZero()) {
                spdlog::warn("[Dev {}] Checkpointed key {} is outside its calculated segment [{}, {}]. Starting from segment start.",
                    actual_device_id, checkpointed_next.toString(16), segment_start_key.toString(16), segment_end_key.toString(16));
            }
        }

        // If after all adjustments, no keys are left for this thread, skip it.
        if (thread_actual_start_key.cmp(segment_end_key) > 0) {
             spdlog::info("[Dev {}] No keys to scan (effective start {} > segment end {}). Skipping.",
                deviceInfo.id, thread_actual_start_key.toString(16), segment_end_key.toString(16));
             next_segment_start_key = segment_end_key + 1; // Prepare for next GPU
             if (next_segment_start_key.isZero() && !segment_end_key.isZero()) { break; } // Overflow
             if (next_segment_start_key.cmp(_config.endKey) > 0 && i < num_gpus -1) { break; } // All keys assigned
             continue;
        }

        RunConfig thread_specific_config = _config; // Copy base config
        thread_specific_config.device = deviceInfo.id; // Set the actual device ID for this thread's config context
                                                       // B, T, P settings from global _config will be used if set,
                                                       // otherwise device_thread_function will apply defaults.

        worker_threads.emplace_back(std::thread(device_thread_function,
                                                deviceInfo,
                                                dev_thread_start_key,
                                                dev_thread_end_key,
                                                thread_specific_config,
                                                std::cref(_config.targets), // Pass targets by const reference
                                                std::cref(_config.targetsFile))); // Pass targetsFile by const reference

        current_batch_start_key = dev_thread_end_key + 1;
        if (current_batch_start_key.isZero() && !dev_thread_end_key.isZero()) {
             Logger::log(LogLevel::Info, "Reached end of addressable keyspace or overflow in key division after device " + std::to_string(deviceInfo.id) + ".");
             break;
        }
        if (current_batch_start_key.cmp(_config.endKey) > 0 && i < num_gpus -1 ) {
            Logger::log(LogLevel::Info, "All keys assigned to GPUs. Remaining " + std::to_string(num_gpus - 1 - i) + " GPUs have no further work.");
            break;
        }
    }

    for (auto& th : worker_threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    Logger::log(LogLevel::Info, "\nAll device threads completed.");
    if(worker_threads.empty() && num_gpus > 0 && total_keys_to_scan > secp256k1::uint256(0)){
        Logger::log(LogLevel::Warning, "No work was dispatched to any GPU threads, though GPUs were specified and keyspace was available. Check keyspace division logic and device parameters.");
    }
    return 0;
}

/**
 * Parses a string in the form of x/y
 */
bool parseShare(const std::string &s, uint32_t &idx, uint32_t &total)
{
    size_t pos = s.find('/');
    if(pos == std::string::npos) {
        return false;
    }

    try {
        idx = util::parseUInt32(s.substr(0, pos));
    } catch(...) {
        return false;
    }

    try {
        total = util::parseUInt32(s.substr(pos + 1));
    } catch(...) {
        return false;
    }

    if(idx == 0 || total == 0) {
        return false;
    }

    if(idx > total) {
        return false;
    }

    return true;
}

int main(int argc, char **argv)
{
	bool optCompressed = false;
	bool optUncompressed = false;
    bool listDevices = false;
    bool optShares = false;
    bool optThreads = false;
    bool optBlocks = false;
    bool optPoints = false;

    uint32_t shareIdx = 0;
    uint32_t numShares = 0;

    // Catch --help first
    for(int i = 1; i < argc; i++) {
        if(std::string(argv[i]) == "--help") {
            usage();
            return 0;
        }
    }

    // Check for supported devices
    try {
        _devices = DeviceManager::getDevices();

        if(_devices.size() == 0) {
            Logger::log(LogLevel::Error, "No devices available");
            return 1;
        }
    } catch(DeviceManager::DeviceManagerException ex) {
        Logger::log(LogLevel::Error, "Error detecting devices: " + ex.msg);
        return 1;
    }

    // Check for arguments
	if(argc == 1) {
		usage();
		return 0;
	}


	CmdParse parser;
	parser.add("-d", "--device", true);
	parser.add("-t", "--threads", true);
	parser.add("-b", "--blocks", true);
	parser.add("-p", "--points", true);
	parser.add("-d", "--device", true);
	parser.add("-c", "--compressed", false);
	parser.add("-u", "--uncompressed", false);
    parser.add("", "--compression", true);
	parser.add("-i", "--in", true);
	parser.add("-o", "--out", true);
    parser.add("-f", "--follow", false);
    parser.add("", "--list-devices", false);
    parser.add("", "--keyspace", true);
    parser.add("", "--continue", true);
    parser.add("", "--share", true);
    parser.add("", "--stride", true);
    parser.add("", "--checkpoint-interval", true);
    parser.add("", "--logfile", true);
    parser.add("", "--loglevel", true);

    // Parse arguments first, then setup logging with parsed values
    try {
        parser.parse(argc, argv);
    } catch(std::string err) {
        // Logger::log(LogLevel::Error, "Error: " + err); // Old
        spdlog::error("Argument parsing error: {}", err); // New
        return 1;
    }

    std::vector<OptArg> args = parser.getArgs();

	for(unsigned int i = 0; i < args.size(); i++) {
		OptArg optArg = args[i];
		std::string opt = args[i].option;

		try {
			if(optArg.equals("-t", "--threads")) {
				_config.threads = util::parseUInt32(optArg.arg);
                optThreads = true;
            } else if(optArg.equals("-b", "--blocks")) {
                _config.blocks = util::parseUInt32(optArg.arg);
                optBlocks = true;
			} else if(optArg.equals("-p", "--points")) {
				_config.pointsPerThread = util::parseUInt32(optArg.arg);
                optPoints = true;
			} else if(optArg.equals("-d", "--device")) {
				// _config.devices parsing logic is here, errors are thrown as std::string
                // No direct std::cout/cerr here, uses throw.
                // Old: //_config.device = util::parseUInt32(optArg.arg);
                // New parsing for comma-separated list
                std::string device_str = optArg.arg;
                std::stringstream ss(device_str); // Requires #include <sstream>
                std::string segment;
                _config.devices.clear(); // Clear any defaults if -d is specified
                while(std::getline(ss, segment, ',')) {
                    if(segment.empty()) continue; // Handle cases like "0,,1" or trailing comma
                    try {
                        // util::trim function might be useful here if stoi has issues with spaces
                        // For now, assume segment is clean or stoi handles it.
                        _config.devices.push_back(std::stoi(segment));
                    } catch (const std::invalid_argument& ia) {
                        throw std::string("Invalid device ID '" + segment + "': not a number.");
                    } catch (const std::out_of_range& oor) {
                        throw std::string("Invalid device ID '" + segment + "': out of range.");
                    }
                }
                if(_config.devices.empty() && !device_str.empty() && device_str.find_first_not_of(',') != std::string::npos){
                     throw std::string("No valid device IDs found in '" + device_str + "'.");
                } else if (device_str.empty() && optArg.isSet) {
                     throw std::string("Device option -d requires an argument (e.g., 0 or 0,1).");
                }
                // Successfully parsed device strings, _config.devices is populated.
			} else if(optArg.equals("-c", "--compressed")) {
				optCompressed = true;
            } else if(optArg.equals("-u", "--uncompressed")) {
                optUncompressed = true;
            } else if(optArg.equals("", "--compression")) {
                _config.compression = parseCompressionString(optArg.arg);
			} else if(optArg.equals("-i", "--in")) {
				_config.targetsFile = optArg.arg;
			} else if(optArg.equals("-o", "--out")) {
				_config.resultsFile = optArg.arg;
            } else if(optArg.equals("", "--list-devices")) {
                listDevices = true;
            } else if(optArg.equals("", "--continue")) {
                _config.checkpointFile = optArg.arg;
            } else if(optArg.equals("", "--keyspace")) {
                secp256k1::uint256 start;
                secp256k1::uint256 end;

                parseKeyspace(optArg.arg, start, end);

                if(start.cmp(secp256k1::N) > 0) {
                    throw std::string("argument is out of range");
                }
                if(start.isZero()) {
                    throw std::string("argument is out of range");
                }

                if(end.cmp(secp256k1::N) > 0) {
                    throw std::string("argument is out of range");
                }

                if(start.cmp(end) > 0) {
                    throw std::string("Invalid argument");
                }

                _config.startKey = start;
                _config.nextKey = start;
                _config.endKey = end;
            } else if(optArg.equals("", "--share")) {
                if(!parseShare(optArg.arg, shareIdx, numShares)) {
                    throw std::string("Invalid argument");
                }
                optShares = true;
            } else if(optArg.equals("", "--stride")) {
                try {
                    _config.stride = secp256k1::uint256(optArg.arg);
                } catch(...) {
                    throw std::string("invalid argument: : expected hex string");
                }

                if(_config.stride.cmp(secp256k1::N) >= 0) {
                    throw std::string("argument is out of range");
                }

                if(_config.stride.cmp(0) == 0) {
                    throw std::string("argument is out of range");
                }
            } else if(optArg.equals("-f", "--follow")) {
                _config.follow = true;
            } else if(optArg.equals("", "--checkpoint-interval")) {
                try {
                    unsigned long long interval_seconds = std::stoull(optArg.arg);
                    if (interval_seconds == 0) {
                        throw std::string("Interval must be a positive integer.");
                    }
                    _config.checkpointInterval = interval_seconds * 1000; // Convert to milliseconds
                } catch (const std::exception& e) {
                    throw std::string("Invalid --checkpoint-interval value '" + optArg.arg + "': " + e.what());
                }
            } else if (optArg.equals("", "--logfile")) {
                g_log_file_path = optArg.arg;
            } else if (optArg.equals("", "--loglevel")) {
                std::string level_str = util::toLower(optArg.arg);
                if (level_str == "trace") g_log_level = spdlog::level::trace;
                else if (level_str == "debug") g_log_level = spdlog::level::debug;
                else if (level_str == "info") g_log_level = spdlog::level::info;
                else if (level_str == "warn" || level_str == "warning") g_log_level = spdlog::level::warn;
                else if (level_str == "error" || level_str == "err") g_log_level = spdlog::level::err; // spdlog uses err
                else if (level_str == "critical") g_log_level = spdlog::level::critical;
                else if (level_str == "off") g_log_level = spdlog::level::off;
                else throw std::string("Invalid log level provided: '" + optArg.arg + "'. Valid levels: trace, debug, info, warn, error, critical, off.");
            }

		} catch(std::string err) {
			// Logger::log(LogLevel::Error, "Error " + opt + ": " + err); // Old
            spdlog::error("Error processing option {}: {}", opt, err); // New
			return 1;
		}
	}

    if(listDevices) {
        // printDeviceList uses printf, which is fine for this specific utility function.
        // No change needed here unless we want to route it through spdlog too.
        printDeviceList(_devices);
        return 0;
    }

	// Device ID validation
    if (parser.found("-d", "--device") && !_config.devices.empty()) {
        for(int devId : _config.devices){
            bool found = false;
            for(const auto& dInfo : _devices){
                if(dInfo.id == devId){
                    found = true;
                    break;
                }
            }
            if(!found){
                 // Logger::log(LogLevel::Error, "Error --device: Specified device ID " + util::format(devId) + " does not exist or is not available."); // Old
                 spdlog::error("Error --device: Specified device ID {} does not exist or is not available.", devId); // New
                 printDeviceList(_devices); // Still uses printf
                 return 1;
            }
        }
    } else if (parser.found("-d", "--device") && _config.devices.empty()){
        // Logger::log(LogLevel::Error, "Error --device: No valid device IDs specified."); // Old
        spdlog::error("Error --device: No valid device IDs specified."); // New
        return 1;
    }


	// Parse operands
	std::vector<std::string> ops = parser.getOperands();

	if(ops.size() == 0) {
		if(_config.targetsFile.length() == 0) {
			// Logger::log(LogLevel::Error, "Missing arguments"); // Old
            spdlog::error("Missing arguments: No target addresses or input file specified."); // New
			usage();
			return 1;
		}
	} else {
		for(unsigned int i = 0; i < ops.size(); i++) {
            if(!Address::verifyAddress(ops[i])) {
                // Logger::log(LogLevel::Error, "Invalid address '" + ops[i] + "'"); // Old
                spdlog::error("Invalid address '{}'", ops[i]); // New
                return 1;
            }
			_config.targets.push_back(ops[i]);
		}
	}
    
    // Calculate where to start and end in the keyspace when the --share option is used
    if(optShares) {
        // Logger::log(LogLevel::Info, "Share " + util::format(shareIdx) + " of " + util::format(numShares)); // Old
        spdlog::info("Share {} of {}", shareIdx, numShares); // New
        secp256k1::uint256 numKeys = _config.endKey - _config.nextKey + 1;

        secp256k1::uint256 diff = numKeys.mod(numShares);
        numKeys = numKeys - diff;

        secp256k1::uint256 shareSize = numKeys.div(numShares);

        secp256k1::uint256 startPos = _config.nextKey + (shareSize * (shareIdx - 1));

        if(shareIdx < numShares) {
            secp256k1::uint256 endPos = _config.nextKey + (shareSize * (shareIdx)) - 1;
            _config.endKey = endPos;
        }

        _config.nextKey = startPos;
        _config.startKey = startPos;
    }

	// Check option for compressed, uncompressed, or both
	if(optCompressed && optUncompressed) {
		_config.compression = PointCompressionType::BOTH;
	} else if(optCompressed) {
		_config.compression = PointCompressionType::COMPRESSED;
	} else if(optUncompressed) {
		_config.compression = PointCompressionType::UNCOMPRESSED;
	}

    if(_config.checkpointFile.length() > 0) {
        readCheckpointFile();
    }

    // Now that arguments are parsed and g_log_level/g_log_file_path are set, initialize logging.
    SetupGlobalLogging();

    // Log some initial info now that logging is fully set up.
    spdlog::info("BitCrack {} starting up...", "vX.Y.Z"); // Replace with actual version later
    spdlog::debug("Log level set to: {}", spdlog::level::to_string_view(g_log_level));
    if(!g_log_file_path.empty()) {
        spdlog::info("Logging to file: {}", g_log_file_path);
    }


    return run();
}