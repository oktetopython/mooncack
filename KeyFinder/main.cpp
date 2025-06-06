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

void writeCheckpoint(secp256k1::uint256 nextKey); // Stays as is for now, using global _config

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
    std::lock_guard<std::mutex> lock(console_mutex);
    // Original resultCallback logic:
    if(_config.resultsFile.length() != 0) {
        // Log with device info if possible, though KeySearchResult doesn't have it directly
        // For now, global log.
		Logger::log(LogLevel::Info, "Found key for address '" + info.address + "'. Written to '" + _config.resultsFile + "'");
		std::string s = info.address + " " + info.privateKey.toString(16) + " " + info.publicKey.toString(info.compressed);
		util::appendToFile(_config.resultsFile, s); // Assuming util::appendToFile is thread-safe or race is acceptable for now
		return;
	}

	std::string logStr = "[Dev: " + info.deviceName + "] Address:     " + info.address + "\n"; // Added deviceName
	logStr += "[Dev: " + info.deviceName + "] Private key: " + info.privateKey.toString(16) + "\n";
	logStr += "[Dev: " + info.deviceName + "] Compressed:  ";

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
    std::lock_guard<std::mutex> lock(console_mutex);
    // Original statusCallback logic:
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
        // For now, it saves the nextKey of the thread that triggered the status.
        Logger::log(LogLevel::Info, "[Dev: " + info.deviceName + "] Checkpoint triggered.");
        writeCheckpoint(info.nextKey); // This saves based on global _config, but with this thread's nextKey.
        _lastUpdate = t; // This update is global, might cause other threads to skip their checkpoint.
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
    Logger::log(LogLevel::Info, devPrefix + devInfo.name + ": Processing keys from " +
                startKey.toString(16) + " to " + endKey.toString(16));

    try {
        // Determine device parameters (use thread_config's values if set, otherwise use defaults for this device)
        DeviceParameters params = getDefaultParameters(devInfo);
        unsigned int actual_blocks = (thread_config.blocks == 0) ? params.blocks : thread_config.blocks;
        unsigned int actual_threads = (thread_config.threads == 0) ? params.threads : thread_config.threads;
        unsigned int actual_points = (thread_config.pointsPerThread == 0) ? params.pointsPerThread : thread_config.pointsPerThread;

        Logger::log(LogLevel::Info, devPrefix + "Using B: " + std::to_string(actual_blocks) +
                   ", T: " + std::to_string(actual_threads) +
                   ", P: " + std::to_string(actual_points));


        KeySearchDevice *d = getDeviceContext(devInfo, actual_blocks, actual_threads, actual_points);
        if (!d) {
            Logger::log(LogLevel::Error, devPrefix + "Failed to get device context.");
            return;
        }

        KeyFinder f(startKey, endKey, thread_config.compression, d, thread_config.stride);

        f.setResultCallback(resultCallback_threadsafe);
        f.setStatusInterval(thread_config.statusInterval);
        f.setStatusCallback(statusCallback_threadsafe);

        f.init();

        if(!targets_file_path.empty()) {
            f.setTargets(targets_file_path);
        } else {
            f.setTargets(targets_list);
        }

        Logger::log(LogLevel::Info, devPrefix + "Starting search...");
        f.run();
        Logger::log(LogLevel::Info, devPrefix + "Search finished.");

        delete d;
    } catch(const KeySearchException &ex) {
        Logger::log(LogLevel::Error, devPrefix + "Error: " + ex.msg);
    } catch(const std::exception &exStd) {
         Logger::log(LogLevel::Error, devPrefix + "Standard Exception: " + exStd.what());
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

void writeCheckpoint(secp256k1::uint256 nextKey)
{
    std::ofstream tmp(_config.checkpointFile, std::ios::out);

    tmp << "start=" << _config.startKey.toString() << std::endl;
    tmp << "next=" << nextKey.toString() << std::endl;
    tmp << "end=" << _config.endKey.toString() << std::endl;
    tmp << "blocks=" << _config.blocks << std::endl;
    tmp << "threads=" << _config.threads << std::endl;
    tmp << "points=" << _config.pointsPerThread << std::endl;
    tmp << "compression=" << getCompressionString(_config.compression) << std::endl;
    // Checkpoint for multiple devices will need rework. For now, save the first device or a comma-separated list.
    if(!_config.devices.empty()){
        tmp << "device=" << _config.devices[0] << std::endl; // Save first device for now
    } else {
        // Potentially save a special value or nothing if no devices were configured (e.g. list-devices run)
        tmp << "device=-1" << std::endl;
    }
    tmp << "elapsed=" << (_config.elapsed + util::getSystemTime() - _startTime) << std::endl;
    tmp << "stride=" << _config.stride.toString();
    tmp.close();
}

void readCheckpointFile()
{
    if(_config.checkpointFile.length() == 0) {
        return;
    }

    ConfigFileReader reader(_config.checkpointFile);

    if(!reader.exists()) {
        return;
    }

    Logger::log(LogLevel::Info, "Loading ' " + _config.checkpointFile + "'");

    std::map<std::string, ConfigFileEntry> entries = reader.read();

    _config.startKey = secp256k1::uint256(entries["start"].value);
    _config.nextKey = secp256k1::uint256(entries["next"].value);
    _config.endKey = secp256k1::uint256(entries["end"].value);

    if(_config.threads == 0 && entries.find("threads") != entries.end()) {
        _config.threads = util::parseUInt32(entries["threads"].value);
    }
    if(_config.blocks == 0 && entries.find("blocks") != entries.end()) {
        _config.blocks = util::parseUInt32(entries["blocks"].value);
    }
    if(_config.pointsPerThread == 0 && entries.find("points") != entries.end()) {
        _config.pointsPerThread = util::parseUInt32(entries["points"].value);
    }
    if(entries.find("compression") != entries.end()) {
        _config.compression = parseCompressionString(entries["compression"].value);
    }
    if(entries.find("elapsed") != entries.end()) {
        _config.elapsed = util::parseUInt32(entries["elapsed"].value);
    }
    if(entries.find("stride") != entries.end()) {
        _config.stride = util::parseUInt64(entries["stride"].value);
    }

    _config.totalkeys = (_config.nextKey - _config.startKey).toUint64();
}

int run()
{
    // Default to device 0 if no devices were specified by user
    if (_config.devices.empty()) {
        if (!_devices.empty() && _devices[0].id >=0) {
            Logger::log(LogLevel::Info, "No device specified via -d, defaulting to device " + std::to_string(_devices[0].id) + " (" + _devices[0].name + ").");
            _config.devices.push_back(_devices[0].id);
        } else {
            Logger::log(LogLevel::Error, "No CUDA/OpenCL devices detected or available to default to.");
            return 1;
        }
    }

    std::string dev_list_str;
    for(size_t i=0; i < _config.devices.size(); ++i) {
        dev_list_str += std::to_string(_config.devices[i]);
        if(i < _config.devices.size() - 1) dev_list_str += ",";
    }
    Logger::log(LogLevel::Info, "Using " + std::to_string(_config.devices.size()) + " GPU(s): " + dev_list_str);

    Logger::log(LogLevel::Info, "Compression: " + getCompressionString(_config.compression));
    Logger::log(LogLevel::Info, "Overall Keyspace Start: " + _config.nextKey.toString(16));
    Logger::log(LogLevel::Info, "Overall Keyspace End:   " + _config.endKey.toString(16));
    Logger::log(LogLevel::Info, "Stride: " + _config.stride.toString());

    _startTime = util::getSystemTime();
    _lastUpdate = _startTime;

    std::vector<std::thread> worker_threads;

    if (_config.nextKey.cmp(_config.endKey) > 0 && !(_config.nextKey == _config.endKey && _config.nextKey == 1)) {
        Logger::log(LogLevel::Error, "Start key is greater than end key. Nothing to scan.");
        return 1;
    }
    secp256k1::uint256 total_keys_to_scan = (_config.endKey - _config.nextKey) + 1;

    unsigned int num_gpus = _config.devices.size();

    if (num_gpus == 0) {
        Logger::log(LogLevel::Error, "Configuration error: No GPUs selected for processing (num_gpus is 0).");
        return 1;
    }
    // Handle case where total_keys_to_scan might be 0 if startKey=endKey but they are not 1 (e.g. single key scan)
    if (total_keys_to_scan.isZero() && !(_config.nextKey == _config.endKey && _config.nextKey == secp256k1::uint256(1))) {
         Logger::log(LogLevel::Info, "Total keys to scan is zero (start equals end, and not 1). Scanning single key: " + _config.nextKey.toString(16));
         total_keys_to_scan = 1; // Ensure single key scan proceeds
    } else if (total_keys_to_scan.isZero()) {
         Logger::log(LogLevel::Info, "Total keys to scan is zero. Nothing to do.");
         return 0;
    }

    secp256k1::uint256 keys_per_gpu_base = total_keys_to_scan / num_gpus;
    secp256k1::uint256 remainder_keys_uint256 = total_keys_to_scan % num_gpus;
    uint64_t remainder_keys = 0;
    if (!remainder_keys_uint256.isZero()) {
        remainder_keys = remainder_keys_uint256.toUint64();
    }

    secp256k1::uint256 current_batch_start_key = _config.nextKey;

    for (unsigned int i = 0; i < num_gpus; ++i) {
        int actual_device_id = _config.devices[i];
        DeviceManager::DeviceInfo deviceInfo;
        bool foundDevInfo = false;
        for(const auto& d : _devices) {
            if(d.id == actual_device_id) {
                deviceInfo = d;
                foundDevInfo = true;
                break;
            }
        }
        if(!foundDevInfo) {
            Logger::log(LogLevel::Error, "Internal Error: Device ID " + std::to_string(actual_device_id) + " not found during thread setup. Should have been caught in main().");
            continue;
        }

        secp256k1::uint256 num_keys_for_this_gpu = keys_per_gpu_base;
        if (i < remainder_keys) {
            num_keys_for_this_gpu = num_keys_for_this_gpu + 1;
        }

        if (num_keys_for_this_gpu.isZero()) {
            Logger::log(LogLevel::Info, "[Dev " + std::to_string(deviceInfo.id) + "] No keys to scan in its share, skipping.");
            continue;
        }

        secp256k1::uint256 dev_thread_start_key = current_batch_start_key;
        secp256k1::uint256 dev_thread_end_key = dev_thread_start_key + num_keys_for_this_gpu - 1;

        if (dev_thread_start_key.cmp(_config.endKey) > 0 && current_batch_start_key.cmp(_config.endKey) <=0 && i > 0) {
             dev_thread_end_key = _config.endKey;
        } else if (dev_thread_end_key.cmp(_config.endKey) > 0) {
             dev_thread_end_key = _config.endKey;
        }

        if (dev_thread_start_key.cmp(dev_thread_end_key) > 0 && !num_keys_for_this_gpu.isZero() ) {
             Logger::log(LogLevel::Info, "[Dev " + std::to_string(deviceInfo.id) + "] No keys to scan due to keyspace adjustments (start=" + dev_thread_start_key.toString(16) + ", end=" + dev_thread_end_key.toString(16) +"). Skipping.");
             // This can happen if total_keys_to_scan is very small compared to num_gpus
             if (current_batch_start_key.cmp(_config.endKey) > 0) break; // All keys assigned
             current_batch_start_key = dev_thread_end_key + 1; // Still advance for next potential GPU
             if (current_batch_start_key.isZero() && !dev_thread_end_key.isZero()) { break; } // Overflow
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

    try {
        parser.parse(argc, argv);
    } catch(std::string err) {
        Logger::log(LogLevel::Error, "Error: " + err);
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
				//_config.device = util::parseUInt32(optArg.arg); // Old single device parsing
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
                    throw std::string("Invalid interval value '" + optArg.arg + "': " + e.what());
                }
            }

		} catch(std::string err) {
			Logger::log(LogLevel::Error, "Error " + opt + ": " + err);
			return 1;
		}
	}

    if(listDevices) {
        printDeviceList(_devices);
        return 0;
    }

	// Device ID validation: if -d was used and devices were parsed, validate them.
    // If -d was not used, _config.devices will be empty, and run() will default it.
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
                 Logger::log(LogLevel::Error, "Error --device: Specified device ID " + util::format(devId) + " does not exist or is not available.");
                 printDeviceList(_devices);
                 return 1;
            }
        }
    } else if (parser.found("-d", "--device") && _config.devices.empty()){
        // This case means -d was given, but no valid IDs were parsed (e.g. -d "" or -d ",," )
        // The parsing logic itself should have thrown an error. If it didn't, this is a fallback.
        Logger::log(LogLevel::Error, "Error --device: No valid device IDs specified.");
        return 1;
    }
    // If !parser.found("-d", "--device"), _config.devices remains empty, run() will handle default.


	// Parse operands
	std::vector<std::string> ops = parser.getOperands();

    // If there are no operands, then we must be reading from a file, otherwise
    // expect addresses on the commandline
	if(ops.size() == 0) {
		if(_config.targetsFile.length() == 0) {
			Logger::log(LogLevel::Error, "Missing arguments");
			usage();
			return 1;
		}
	} else {
		for(unsigned int i = 0; i < ops.size(); i++) {
            if(!Address::verifyAddress(ops[i])) {
                Logger::log(LogLevel::Error, "Invalid address '" + ops[i] + "'");
                return 1;
            }
			_config.targets.push_back(ops[i]);
		}
	}
    
    // Calculate where to start and end in the keyspace when the --share option is used
    if(optShares) {
        Logger::log(LogLevel::Info, "Share " + util::format(shareIdx) + " of " + util::format(numShares));
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

    return run();
}