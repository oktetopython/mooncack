#ifndef _KEY_FINDER_H
#define _KEY_FINDER_H

#include <stdint.h>
#include <vector>
#include <set>
#include <atomic> // Required for std::atomic<bool>
#include "secp256k1.h"
#include "KeySearchTypes.h"
#include "KeySearchDevice.h"


class KeyFinder {

private:

    KeySearchDevice *_device;

	unsigned int _compression;

	std::set<KeySearchTarget> _targets;

	uint64_t _statusInterval;

    secp256k1::uint256 _stride = 1;
	uint64_t _iterCount;
	uint64_t _total;
	uint64_t _totalTime;

    secp256k1::uint256 _startKey;
    secp256k1::uint256 _endKey;

	// Each index of each thread gets a flag to indicate if it found a valid hash
	bool _running; // Existing running flag
    std::atomic<bool>* _stopFlagPtr; // New atomic stop flag pointer for C API control

	void(*_resultCallback)(KeySearchResult);
	void(*_statusCallback)(KeySearchStatus);


	static void defaultResultCallback(KeySearchResult result);
	static void defaultStatusCallback(KeySearchStatus status);

	void removeTargetFromList(const unsigned int value[5]);
	bool isTargetInList(const unsigned int value[5]);
	void setTargetsOnDevice();

public:

    KeyFinder(const secp256k1::uint256 &startKey, const secp256k1::uint256 &endKey, int compression, KeySearchDevice* device, const secp256k1::uint256 &stride);

	~KeyFinder();

	void init();
	void run();
	void stop();

	void setResultCallback(void(*callback)(KeySearchResult));
	void setStatusCallback(void(*callback)(KeySearchStatus));
	void setStatusInterval(uint64_t interval);

    void setStopFlag(std::atomic<bool>* flag); // New method to set the stop flag

	void setTargets(std::string targetFile);
	void setTargets(std::vector<std::string> &targets);

    secp256k1::uint256 getNextKey();
};

#endif