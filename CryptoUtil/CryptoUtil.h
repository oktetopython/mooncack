#ifndef _CRYPTO_UTIL_H

namespace crypto {

	class Rng {
		unsigned int _state[16];
		unsigned int _counter;

		void reseed();

	public:
		Rng();
		void get(unsigned char *buf, int len);
	};


	void ripemd160(unsigned int *msg, unsigned int *digest);

	void sha256Init(unsigned int *digest);
	void sha256(unsigned int *msg, unsigned int *digest); // Operates on uint arrays

	// New functions for raw byte arrays
	void sha256_raw(const unsigned char* data, size_t len, unsigned char out_hash[32]);
	void sha256_double_raw(const unsigned char* data, size_t len, unsigned char out_hash[32]);

	unsigned int checksum(const unsigned int *hash); // This is likely for the old address generation, may not be Base58Check checksum
};

#endif