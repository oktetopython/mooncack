#include "AddressUtil.h"
#include "CryptoUtil.h"

#include <stdio.h>
#include <string.h>

static unsigned int endian(unsigned int x)
{
	return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

bool Address::verifyAddress(std::string address)
{
    // Basic length sanity checks (min length for version + hash160 + checksum)
    // P2PKH: 1 (version) + 20 (hash160) + 4 (checksum) = 25 bytes decoded. Encoded can be 26-35 chars.
    if (address.length() < 26 || address.length() > 35) { // Common P2PKH/P2SH lengths
        return false;
    }

    std::vector<unsigned char> full_payload_with_version;
    if (!Base58::Base58CheckDecode(address, full_payload_with_version)) {
        return false; // Base58Check decoding failed (checksum mismatch or invalid chars)
    }

    // Check payload length (1 byte version + 20 bytes HASH160)
    if (full_payload_with_version.size() != 21) {
        // This check is specific to addresses that embed a 20-byte HASH160.
        // Other address types (like WIF private keys, BIP32 extended keys) will have different lengths
        // and version bytes. For now, we focus on P2PKH/P2SH-like HASH160 extraction.
        return false;
    }

    unsigned char version_byte = full_payload_with_version[0];

    // Validate known Bitcoin mainnet version bytes
    // 0x00 for P2PKH (addresses starting with '1')
    // 0x05 for P2SH (addresses starting with '3')
    // Testnet versions are different (e.g., 0x6F for P2PKH, 0xC4 for P2SH)
    // This verifier could be extended to support testnet or other coin types.
    if (version_byte == 0x00 || version_byte == 0x05) {
        // Could add more specific checks based on the first char of 'address' vs version_byte if desired
        // e.g. version 0x00 should start with '1', version 0x05 with '3' on mainnet.
        // Base58ToBytes itself doesn't know about expected first char for a version.
        return true;
    }

    // Add other known version bytes if needed, or make this configurable.
    // For now, only mainnet P2PKH and P2SH are considered "valid" by this function.

    return false; // Unknown/unsupported version byte
}

std::string Address::fromPublicKey(const secp256k1::ecpoint &p, bool compressed)
{
	unsigned int xWords[8] = { 0 };
	unsigned int yWords[8] = { 0 };

	p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	unsigned int digest[5];

	if(compressed) {
		Hash::hashPublicKeyCompressed(xWords, yWords, digest);
	} else {
		Hash::hashPublicKey(xWords, yWords, digest);
	}

	unsigned int checksum = crypto::checksum(digest);

	unsigned int addressWords[8] = { 0 };
	for(int i = 0; i < 5; i++) {
		addressWords[2 + i] = digest[i];
	}
	addressWords[7] = checksum;

	secp256k1::uint256 addressBigInt(addressWords, secp256k1::uint256::BigEndian);

	return "1" + Base58::toBase58(addressBigInt);
}

void Hash::hashPublicKey(const secp256k1::ecpoint &p, unsigned int *digest)
{
	unsigned int xWords[8];
	unsigned int yWords[8];

	p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	hashPublicKey(xWords, yWords, digest);
}


void Hash::hashPublicKeyCompressed(const secp256k1::ecpoint &p, unsigned int *digest)
{
	unsigned int xWords[8];
	unsigned int yWords[8];

	p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	hashPublicKeyCompressed(xWords, yWords, digest);
}

void Hash::hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digest)
{
	unsigned int msg[16];
	unsigned int sha256Digest[8];

	// 0x04 || x || y
	msg[15] = (y[7] >> 8) | (y[6] << 24);
	msg[14] = (y[6] >> 8) | (y[5] << 24);
	msg[13] = (y[5] >> 8) | (y[4] << 24);
	msg[12] = (y[4] >> 8) | (y[3] << 24);
	msg[11] = (y[3] >> 8) | (y[2] << 24);
	msg[10] = (y[2] >> 8) | (y[1] << 24);
	msg[9] = (y[1] >> 8) | (y[0] << 24);
	msg[8] = (y[0] >> 8) | (x[7] << 24);
	msg[7] = (x[7] >> 8) | (x[6] << 24);
	msg[6] = (x[6] >> 8) | (x[5] << 24);
	msg[5] = (x[5] >> 8) | (x[4] << 24);
	msg[4] = (x[4] >> 8) | (x[3] << 24);
	msg[3] = (x[3] >> 8) | (x[2] << 24);
	msg[2] = (x[2] >> 8) | (x[1] << 24);
	msg[1] = (x[1] >> 8) | (x[0] << 24);
	msg[0] = (x[0] >> 8) | 0x04000000;


	crypto::sha256Init(sha256Digest);
	crypto::sha256(msg, sha256Digest);

	// Zero out the message
	for(int i = 0; i < 16; i++) {
		msg[i] = 0;
	}

	// Set first byte, padding, and length
	msg[0] = (y[7] << 24) | 0x00800000;
	msg[15] = 65 * 8;

	crypto::sha256(msg, sha256Digest);

	for(int i = 0; i < 16; i++) {
		msg[i] = 0;
	}

	// Swap to little-endian
	for(int i = 0; i < 8; i++) {
		msg[i] = endian(sha256Digest[i]);
	}

	// Message length, little endian
	msg[8] = 0x00000080;
	msg[14] = 256;
	msg[15] = 0;

	crypto::ripemd160(msg, digest);
}



void Hash::hashPublicKeyCompressed(const unsigned int *x, const unsigned int *y, unsigned int *digest)
{
	unsigned int msg[16] = { 0 };
	unsigned int sha256Digest[8];

	// Compressed public key format
	msg[15] = 33 * 8;

	msg[8] = (x[7] << 24) | 0x00800000;
	msg[7] = (x[7] >> 8) | (x[6] << 24);
	msg[6] = (x[6] >> 8) | (x[5] << 24);
	msg[5] = (x[5] >> 8) | (x[4] << 24);
	msg[4] = (x[4] >> 8) | (x[3] << 24);
	msg[3] = (x[3] >> 8) | (x[2] << 24);
	msg[2] = (x[2] >> 8) | (x[1] << 24);
	msg[1] = (x[1] >> 8) | (x[0] << 24);

	if(y[7] & 0x01) {
		msg[0] = (x[0] >> 8) | 0x03000000;
	} else {
		msg[0] = (x[0] >> 8) | 0x02000000;
	}

	crypto::sha256Init(sha256Digest);
	crypto::sha256(msg, sha256Digest);

	for(int i = 0; i < 16; i++) {
		msg[i] = 0;
	}

	// Swap to little-endian
	for(int i = 0; i < 8; i++) {
		msg[i] = endian(sha256Digest[i]);
	}

	// Message length, little endian
	msg[8] = 0x00000080;
	msg[14] = 256;
	msg[15] = 0;

	crypto::ripemd160(msg, digest);
}