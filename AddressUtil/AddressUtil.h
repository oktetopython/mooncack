#ifndef _ADDRESS_UTIL_H
#define _ADDRESS_UTIL_H

#include "secp256k1.h"

namespace Address {
	std::string fromPublicKey(const secp256k1::ecpoint &p, bool compressed = false);
	bool verifyAddress(std::string address);
};

namespace Base58 {
	std::string toBase58(const secp256k1::uint256 &x);
	secp256k1::uint256 toBigInt(const std::string &s);
	void getMinMaxFromPrefix(const std::string &prefix, secp256k1::uint256 &minValueOut, secp256k1::uint256 &maxValueOut);

	void toHash160(const std::string &s, unsigned int hash[5]); // This will be updated to use Base58CheckDecode

	bool isBase58(std::string s);

	// Decodes a Base58 string to a byte vector, preserving leading zeros.
	std::vector<unsigned char> Base58ToBytes(const std::string& base58_input);

	// Performs Base58Check decoding. Output includes version byte(s) and payload.
	// Returns true on success, false if checksum fails or input is invalid.
	bool Base58CheckDecode(const std::string& base58_input, std::vector<unsigned char>& out_full_payload);
};



namespace Hash {


	void hashPublicKey(const secp256k1::ecpoint &p, unsigned int *digest);
	void hashPublicKeyCompressed(const secp256k1::ecpoint &p, unsigned int *digest);

	void hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digest);
	void hashPublicKeyCompressed(const unsigned int *x, const unsigned int *y, unsigned int *digest);

};


#endif