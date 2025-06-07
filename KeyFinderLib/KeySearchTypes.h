#ifndef _KEY_FINDER_TYPES
#define _KEY_FINDER_TYPES

#include <stdint.h>
#include <string>
#include <functional> // For std::hash
#include <cstring>    // For memcmp
#include "secp256k1.h"

namespace PointCompressionType {
    enum Value {
        COMPRESSED = 0,
        UNCOMPRESSED = 1,
        BOTH = 2
    };
}

typedef struct hash160 {

    unsigned int h[5];

    hash160(const unsigned int hash[5])
    {
        memcpy(this->h, hash, sizeof(unsigned int) * 5);
    }

    // Default constructor for unordered_set
    hash160()
    {
        memset(this->h, 0, sizeof(unsigned int) * 5);
    }

    bool operator==(const hash160& other) const {
        return memcmp(h, other.h, sizeof(h)) == 0;
    }
}hash160;

// Hash functor for hash160
struct Hash160Hasher {
    std::size_t operator()(const hash160& val) const {
        std::size_t seed = 0;
        // Simple hash combination.
        for(int i = 0; i < 5; ++i) {
            seed ^= std::hash<unsigned int>()(val.h[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};


typedef struct {
    int device;
    double speed;
    uint64_t total;
    uint64_t totalTime;
    std::string deviceName;
    uint64_t freeMemory;
    uint64_t deviceMemory;
    uint64_t targets;
    secp256k1::uint256 nextKey;
    int physicalDeviceId; // Added for multi-GPU progress tracking
}KeySearchStatus;


class KeySearchTarget {

public:
    unsigned int value[5];

    KeySearchTarget()
    {
        memset(value, 0, sizeof(value));
    }

    KeySearchTarget(const unsigned int h[5])
    {
        for(int i = 0; i < 5; i++) {
            value[i] = h[i];
        }
    }


    bool operator==(const KeySearchTarget &t) const
    {
        for(int i = 0; i < 5; i++) {
            if(value[i] != t.value[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator<(const KeySearchTarget &t) const
    {
        for(int i = 0; i < 5; i++) {
            if(value[i] < t.value[i]) {
                return true;
            } else if(value[i] > t.value[i]) {
                return false;
            }
        }

        return false;
    }

    bool operator>(const KeySearchTarget &t) const
    {
        for(int i = 0; i < 5; i++) {
            if(value[i] > t.value[i]) {
                return true;
            } else if(value[i] < t.value[i]) {
                return false;
            }
        }

        return false;
    }
};

#endif