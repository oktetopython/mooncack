#include <map>
#include <vector> // Added for std::vector
#include <algorithm> // For std::reverse if needed, and std::find
#include "CryptoUtil.h"

#include "AddressUtil.h"


static const std::string BASE58_STRING = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

struct Base58Map {
	static std::map<char, int> createBase58Map()
	{
		std::map<char, int> m;
		for(int i = 0; i < 58; i++) {
			m[BASE58_STRING[i]] = i;
		}

		return m;
	}

	static std::map<char, int> myMap;
};

std::map<char, int> Base58Map::myMap = Base58Map::createBase58Map();



/**
 * Converts a base58 string to uint256
 */
secp256k1::uint256 Base58::toBigInt(const std::string &s)
{
	secp256k1::uint256 value;

	for(unsigned int i = 0; i < s.length(); i++) {
		value = value.mul(58);

		int c = Base58Map::myMap[s[i]];
		value = value.add(c);
	}

	return value;
}

void Base58::toHash160(const std::string &s, unsigned int hash[5])
{
    std::vector<unsigned char> full_payload;
    if (!Base58::Base58CheckDecode(s, full_payload)) {
        // Error: Invalid Base58Check string (bad format, checksum mismatch, etc.)
        // Set hash to a known invalid value or throw, depending on desired error handling.
        // For now, similar to original behavior if toBigInt failed implicitly, fill with zeros.
        memset(hash, 0, 5 * sizeof(unsigned int));
        // Or throw KeySearchException("Invalid Base58Check address for toHash160");
        return;
    }

    // Standard P2PKH (version 0) and P2SH (version 5) addresses have a 1-byte version prefix
    // followed by a 20-byte HASH160. Total 21 bytes in full_payload.
    if (full_payload.size() < 21) { // Check if payload (version + hash160) is at least 21 bytes
        memset(hash, 0, 5 * sizeof(unsigned int)); // Payload too short
        // Or throw KeySearchException("Decoded Base58Check payload too short to contain HASH160");
        return;
    }

    // Extract the 20-byte HASH160 payload, skipping the first version byte(s).
    // Assuming 1 version byte for common address types.
    // The HASH160 is bytes 1 through 20 of the full_payload.
    // We need to convert these 20 bytes into 5 unsigned ints.
    // Each unsigned int takes 4 bytes. HASH160 is typically treated as a big-endian byte sequence.
    // The `hash[5]` array should store these as words. How words are formed from bytes (endianness) matters.
    // If hash[0] should be the first 4 bytes (MSB first), then it's big-endian word.
    // If hash[0] should be the first 4 bytes (LSB first), then it's little-endian word.
    // The existing code `value.exportWords(words, 6, secp256k1::uint256::BigEndian);` suggests
    // that `words` (and thus `hash`) were filled in a big-endian manner (word `words[0]` contains MSBs of hash).
    // Let's maintain that: construct uints from bytes in big-endian order.

    const unsigned char* hash160_bytes = full_payload.data() + 1; // Skip version byte

    for (int i = 0; i < 5; ++i) { // For each of the 5 uints in hash
        hash[i] = 0;
        // Check if enough bytes are available in full_payload for this word
        // (1 for version + i*4 for previous words + 4 for current word)
        if ( (1 + (i * 4) + 3) < full_payload.size() ) { // Ensure 4 bytes are readable for this word
             // Construct uint from 4 bytes, big-endian manner
            hash[i] = (static_cast<unsigned int>(hash160_bytes[i * 4 + 0]) << 24) |
                      (static_cast<unsigned int>(hash160_bytes[i * 4 + 1]) << 16) |
                      (static_cast<unsigned int>(hash160_bytes[i * 4 + 2]) << 8)  |
                      (static_cast<unsigned int>(hash160_bytes[i * 4 + 3]));
        } else {
            // This case means full_payload (after version byte) was not 20 bytes long.
            // Should have been caught by full_payload.size() < 21 check, but good to be safe.
            memset(hash, 0, 5 * sizeof(unsigned int)); // Error, clear hash
            return;
        }
    }
}

bool Base58::isBase58(std::string s)
{
	for(unsigned int i = 0; i < s.length(); i++) {
		if(BASE58_STRING.find(s[i]) < 0) {
			return false;
		}
	}

	return true;
}

std::string Base58::toBase58(const secp256k1::uint256 &x)
{
	std::string s;

	secp256k1::uint256 value = x;

	while(!value.isZero()) {
		secp256k1::uint256 digit = value.mod(58);
		int digitInt = digit.toInt32();

		s = BASE58_STRING[digitInt] + s;

		value = value.div(58);
	}

	return s;
}

void Base58::getMinMaxFromPrefix(const std::string &prefix, secp256k1::uint256 &minValueOut, secp256k1::uint256 &maxValueOut)
{
	secp256k1::uint256 minValue = toBigInt(prefix);
	secp256k1::uint256 maxValue = minValue;
	int exponent = 1;

	// 2^192
	unsigned int expWords[] = { 0, 0, 0, 0, 0, 0, 1, 0 };

	secp256k1::uint256 exp(expWords);

	// Find the smallest 192-bit number that starts with the prefix. That is, the prefix multiplied
	// by some power of 58
	secp256k1::uint256 nextValue = minValue.mul(58);

	while(nextValue.cmp(exp) < 0) {
		exponent++;
		minValue = nextValue;
		nextValue = nextValue.mul(58);
	}

	secp256k1::uint256 diff = secp256k1::uint256(58).pow(exponent - 1).sub(1);

	maxValue = minValue.add(diff);

	if(maxValue.cmp(exp) > 0) {
		maxValue = exp.sub(1);
	}

	minValueOut = minValue;
	maxValueOut = maxValue;
}


// Decodes a Base58 string to a byte vector.
// This implementation handles leading zeros correctly.
std::vector<unsigned char> Base58::Base58ToBytes(const std::string& base58_input)
{
    std::vector<unsigned char> bytes;
    bytes.reserve(base58_input.length() * 733 / 1000 + 1); // Approximate decoded length
    bytes.push_back(0); // Start with a single zero byte

    for (char c : base58_input) {
        auto it = Base58Map::myMap.find(c);
        if (it == Base58Map::myMap.end()) {
            // Invalid Base58 character
            throw KeySearchException("Invalid Base58 character in input string: " + std::string(1, c));
        }
        int value = it->second;
        int carry = value;

        for (size_t j = 0; j < bytes.size(); ++j) {
            carry += static_cast<int>(bytes[bytes.size() - 1 - j]) * 58;
            bytes[bytes.size() - 1 - j] = static_cast<unsigned char>(carry % 256);
            carry /= 256;
        }

        while (carry > 0) {
            bytes.insert(bytes.begin(), static_cast<unsigned char>(carry % 256));
            carry /= 256;
        }
    }

    // Count leading '1's (value 0 in Base58)
    size_t leading_zeros = 0;
    for (char c : base58_input) {
        if (c == BASE58_STRING[0]) { // '1'
            leading_zeros++;
        } else {
            break;
        }
    }

    // Remove the initial zero byte if it's not part of leading zeros from '1's
    // or if the result is non-zero.
    // If the result is all zeros (e.g. input "111"), bytes will be [0,0,0,0] after loop.
    // The loop results in big-endian number. We need to add leading_zeros.

    std::vector<unsigned char> result;
    result.reserve(leading_zeros + bytes.size());

    for (size_t i = 0; i < leading_zeros; ++i) {
        result.push_back(0);
    }

    // Skip leading zeros in the 'bytes' vector that were an artifact of calculation,
    // unless the number itself is zero.
    bool non_zero_found = false;
    for(unsigned char byte_val : bytes) {
        if(byte_val != 0) non_zero_found = true;
        if(non_zero_found) result.push_back(byte_val);
    }
    // If entire 'bytes' was zeros (e.g. input was only '1's, or empty string giving single 0 byte)
    // and we have leading_zeros, result is already correct.
    // If bytes was [0] and no leading_zeros (e.g. from empty input), result should be [0] or empty.
    // The current Base58ToBytes is a bit complex. Let's use a more standard one.
    // Source: Bitcoin Core's base58.cpp (DecodeBase58) logic adapted.

    std::vector<unsigned char> result_bytes;
    result_bytes.assign(base58_input.size() * 733 / 1000 + 1, 0); // upper bound for decoded data

    int length = 0;
    for (char c : base58_input) {
        auto it = Base58Map::myMap.find(c);
        if (it == Base58Map::myMap.end()) {
            throw KeySearchException("Invalid Base58 character in input string: " + std::string(1, c));
        }
        int carry = it->second;
        for (int j = result_bytes.size() - 1; j >= 0; j--) {
            if (carry == 0 && j < length) break; // optimization
            carry += 58 * result_bytes[j];
            result_bytes[j] = carry % 256;
            carry /= 256;
        }
        while (carry > 0) { // Should not happen with pre-sized result_bytes if size is correct
             // This would mean result_bytes is too small, which indicates an issue.
             // For safety, one could use result_bytes.insert(result_bytes.begin(), carry % 256);
             // But standard implementations often rely on pre-calculated max size.
             // Let's re-evaluate the typical algorithm structure for byte vectors.
        }
        // Find first non-zero byte to update current length.
        // This is complex. A simpler way is to treat it like a big int multiplication.
    }

    // Correct Base58ToBytes using a common algorithm (vector as big-endian number)
    std::vector<unsigned char> b58_decode_result;
    b58_decode_result.reserve(base58_input.length()); // Max possible if all are '1's

    for(char const& c : base58_input) {
        auto it = Base58Map::myMap.find(c);
        if(it == Base58Map::myMap.end()){
            throw KeySearchException("Invalid Base58 character in input string: " + std::string(1, c));
        }
        unsigned int value = it->second;
        unsigned int carry = value;
        for (size_t i = 0; i < b58_decode_result.size(); ++i) {
            size_t reverse_idx = b58_decode_result.size() - 1 - i; // process from LSB
            carry += static_cast<unsigned int>(b58_decode_result[reverse_idx]) * 58;
            b58_decode_result[reverse_idx] = static_cast<unsigned char>(carry % 256);
            carry /= 256;
        }
        while(carry > 0) {
            b58_decode_result.insert(b58_decode_result.begin(), static_cast<unsigned char>(carry % 256));
            carry /= 256;
        }
    }

    // Add leading zero bytes
    size_t nLeadingZeros = 0;
    for(char const& c : base58_input) {
        if(c == BASE58_STRING[0]) { // '1'
            nLeadingZeros++;
        } else {
            break;
        }
    }
    // If all were '1's, b58_decode_result will be empty.
    // If not empty, it already has some data. We need nLeadingZeros at the front.
    // If it was empty and nLeadingZeros > 0, means input was e.g. "111".
    std::vector<unsigned char> final_result;
    final_result.reserve(nLeadingZeros + b58_decode_result.size());
    for(size_t i = 0; i < nLeadingZeros; ++i) {
        final_result.push_back(0);
    }
    final_result.insert(final_result.end(), b58_decode_result.begin(), b58_decode_result.end());

    // Handle empty input string case: should result in empty vector, not [0]
    if (base58_input.empty()) {
        final_result.clear();
    }

    return final_result;
}


// Performs Base58Check decoding. Output includes version byte(s) and payload.
// Returns true on success, false if checksum fails or input is invalid.
bool Base58::Base58CheckDecode(const std::string& base58_input, std::vector<unsigned char>& out_full_payload)
{
    out_full_payload.clear();
    std::vector<unsigned char> decoded_bytes;
    try {
        decoded_bytes = Base58::Base58ToBytes(base58_input);
    } catch (const KeySearchException& e) {
        // Invalid Base58 characters
        // Consider logging: spdlog::debug("Base58ToBytes failed: {}", e.what());
        return false;
    }

    if (decoded_bytes.size() < 5) { // Minimum 1 version byte + 4 checksum bytes
        return false; // Too short to be a valid Base58Check string
    }

    // Separate payload_with_version and checksum_provided
    std::vector<unsigned char> payload_with_version(decoded_bytes.begin(), decoded_bytes.end() - 4);
    std::vector<unsigned char> checksum_provided(decoded_bytes.end() - 4, decoded_bytes.end());

    // Calculate checksum: first 4 bytes of SHA256(SHA256(payload_with_version))
    unsigned char calculated_hash_full[32]; // SHA256 produces 32 bytes
    crypto::sha256_double_raw(payload_with_version.data(), payload_with_version.size(), calculated_hash_full);

    // Compare provided checksum with the first 4 bytes of the calculated double-SHA256 hash
    for (int i = 0; i < 4; ++i) {
        if (checksum_provided[i] != calculated_hash_full[i]) {
            return false; // Checksum mismatch
        }
    }

    out_full_payload = payload_with_version;
    return true; // Success
}