#ifndef _SECP256K1_CUH
#define _SECP256K1_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "ptx.cuh"


/**
 Prime modulus 2^256 - 2^32 - 977
 */
__constant__ static unsigned int _P[8] = {
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

/**
 Base point X
 */
__constant__ static unsigned int _GX[8] = {
	0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};


/**
 Base point Y
 */
__constant__ static unsigned int _GY[8] = {
	0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};


/**
 * Group order
 */
__constant__ static unsigned int _N[8] = {
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

__constant__ static unsigned int _BETA[8] = {
	0x7AE96A2B, 0x657C0710, 0x6E64479E, 0xAC3434E9, 0x9CF04975, 0x12F58995, 0xC1396C28, 0x719501EE
};


__constant__ static unsigned int _LAMBDA[8] = {
	0x5363AD4C, 0xC05C30E0, 0xA5261C02, 0x8812645A, 0x122E22EA, 0x20816678, 0xDF02967C, 0x1B23BD72
};

// For GPU-side fixed-window k*G, window size w=4 (16 entries)
// Each point coordinate (X or Y) is 8 uints (256-bit)
// Total size = 16 entries * 8 uints/entry = 128 uints per table
__constant__ unsigned int c_gWindowTableX[16 * 8];
__constant__ unsigned int c_gWindowTableY[16 * 8];

// Forward declare for use in point operations
__device__ static void invModP(unsigned int value[8]);
__device__ static void mulModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8]);
__device__ static void addModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8]);
__device__ static void subModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8]);
__device__ __forceinline__ static void copyBigInt(const unsigned int src[8], unsigned int dest[8]);
__device__ __forceinline__ bool isInfinity(const unsigned int x[8]);
__device__ static bool equal(const unsigned int *a, const unsigned int *b);


// Device function for point doubling: (xR, yR) = 2 * (xP, yP)
__device__ static void device_doublePoint(const unsigned int xP[8], const unsigned int yP[8], unsigned int xR[8], unsigned int yR[8]) {
    if (isInfinity(yP) || (yP[0]==0 && yP[1]==0 && yP[2]==0 && yP[3]==0 && yP[4]==0 && yP[5]==0 && yP[6]==0 && yP[7]==0) ) { // Check if yP is zero effectively
        // Point is at infinity or has y=0 (2*P = infinity)
        for(int i=0; i<8; ++i) xR[i] = yR[i] = 0xFFFFFFFF;
        return;
    }

    unsigned int lambda[8], temp[8], yP2[8];

    // lambda = (3 * xP^2) * (2 * yP)^-1 mod p
    // 2 * yP
    addModP(yP, yP, yP2);
    invModP(yP2); // yP2 is now (2*yP)^-1

    // 3 * xP^2
    mulModP(xP, xP, temp);    // xP^2
    addModP(temp, temp, lambda); // 2 * xP^2
    addModP(lambda, temp, lambda); // 3 * xP^2

    mulModP(lambda, yP2, lambda); // lambda = (3 * xP^2) * (2*yP)^-1

    // xR = lambda^2 - 2 * xP mod p
    mulModP(lambda, lambda, xR); // lambda^2
    subModP(xR, xP, xR);       // lambda^2 - xP
    subModP(xR, xP, xR);       // lambda^2 - 2*xP

    // yR = lambda * (xP - xR) - yP mod p
    subModP(xP, xR, temp);       // xP - xR
    mulModP(lambda, temp, yR);   // lambda * (xP - xR)
    subModP(yR, yP, yR);       // lambda * (xP - xR) - yP
}

// Device function for point addition: (xR, yR) = (xP1, yP1) + (xP2, yP2)
__device__ static void device_addPoints(const unsigned int xP1[8], const unsigned int yP1[8],
                                     const unsigned int xP2[8], const unsigned int yP2[8],
                                     unsigned int xR[8], unsigned int yR[8]) {
    if (isInfinity(xP1)) {
        copyBigInt(xP2, xR);
        copyBigInt(yP2, yR);
        return;
    }
    if (isInfinity(xP2)) {
        copyBigInt(xP1, xR);
        copyBigInt(yP1, yR);
        return;
    }

    if (equal(xP1, xP2)) {
        if (equal(yP1, yP2)) {
            // P1 == P2, use double point
            device_doublePoint(xP1, yP1, xR, yR);
        } else {
            // P1 == -P2 (x1=x2, y1!=y2 implies y1 = -y2 mod p), result is point at infinity
            for(int i=0; i<8; ++i) xR[i] = yR[i] = 0xFFFFFFFF;
        }
        return;
    }

    unsigned int lambda[8], temp[8];

    // lambda = (yP2 - yP1) * (xP2 - xP1)^-1 mod p
    subModP(yP2, yP1, lambda); // yP2 - yP1
    subModP(xP2, xP1, temp);   // xP2 - xP1
    invModP(temp);             // (xP2 - xP1)^-1
    mulModP(lambda, temp, lambda); // lambda calculation complete

    // xR = lambda^2 - xP1 - xP2 mod p
    mulModP(lambda, lambda, xR); // lambda^2
    subModP(xR, xP1, xR);      // lambda^2 - xP1
    subModP(xR, xP2, xR);      // lambda^2 - xP1 - xP2

    // yR = lambda * (xP1 - xR) - yP1 mod p
    subModP(xP1, xR, temp);      // xP1 - xR
    mulModP(lambda, temp, yR);   // lambda * (xP1 - xR)
    subModP(yR, yP1, yR);      // lambda * (xP1 - xR) - yP1
}


__device__ __forceinline__ bool isInfinity(const unsigned int x[8])
{
	bool isf = true;

	for(int i = 0; i < 8; i++) {
		if(x[i] != 0xffffffff) {
			isf = false;
		}
	}

	return isf;
}

__device__ __forceinline__ static void copyBigInt(const unsigned int src[8], unsigned int dest[8])
{
	for(int i = 0; i < 8; i++) {
		dest[i] = src[i];
	}
}

__device__ static bool equal(const unsigned int *a, const unsigned int *b)
{
	bool eq = true;

	for(int i = 0; i < 8; i++) {
		eq &= (a[i] == b[i]);
	}

	return eq;
}

/**
 * Reads an 8-word big integer from device memory
 */
__device__ static void readInt(const unsigned int *ara, int idx, unsigned int x[8])
{
	int totalThreads = gridDim.x * blockDim.x;

	int base = idx * totalThreads * 8;

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	int index = base + threadId;

	for (int i = 0; i < 8; i++) {
		x[i] = ara[index];
		index += totalThreads;
	}
}

__device__ static unsigned int readIntLSW(const unsigned int *ara, int idx)
{
	int totalThreads = gridDim.x * blockDim.x;

	int base = idx * totalThreads * 8;

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	int index = base + threadId;

	return ara[index + totalThreads * 7];
}

/**
 * Writes an 8-word big integer to device memory
 */
__device__ static void writeInt(unsigned int *ara, int idx, const unsigned int x[8])
{
	int totalThreads = gridDim.x * blockDim.x;

	int base = idx * totalThreads * 8;

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	int index = base + threadId;

	for (int i = 0; i < 8; i++) {
		ara[index] = x[i];
		index += totalThreads;
	}
}

/**
 * Subtraction mod p
 */
__device__ static void subModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	sub_cc(c[7], a[7], b[7]);
	subc_cc(c[6], a[6], b[6]);
	subc_cc(c[5], a[5], b[5]);
	subc_cc(c[4], a[4], b[4]);
	subc_cc(c[3], a[3], b[3]);
	subc_cc(c[2], a[2], b[2]);
	subc_cc(c[1], a[1], b[1]);
	subc_cc(c[0], a[0], b[0]);

	unsigned int borrow = 0;
	subc(borrow, 0, 0);

	if (borrow) {
		add_cc(c[7], c[7], _P[7]);
		addc_cc(c[6], c[6], _P[6]);
		addc_cc(c[5], c[5], _P[5]);
		addc_cc(c[4], c[4], _P[4]);
		addc_cc(c[3], c[3], _P[3]);
		addc_cc(c[2], c[2], _P[2]);
		addc_cc(c[1], c[1], _P[1]);
		addc(c[0], c[0], _P[0]);
	}
}

__device__ static unsigned int add(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	add_cc(c[7], a[7], b[7]);
	addc_cc(c[6], a[6], b[6]);
	addc_cc(c[5], a[5], b[5]);
	addc_cc(c[4], a[4], b[4]);
	addc_cc(c[3], a[3], b[3]);
	addc_cc(c[2], a[2], b[2]);
	addc_cc(c[1], a[1], b[1]);
	addc_cc(c[0], a[0], b[0]);

	unsigned int carry = 0;
	addc(carry, 0, 0);

	return carry;
}

__device__ static unsigned int sub(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	sub_cc(c[7], a[7], b[7]);
	subc_cc(c[6], a[6], b[6]);
	subc_cc(c[5], a[5], b[5]);
	subc_cc(c[4], a[4], b[4]);
	subc_cc(c[3], a[3], b[3]);
	subc_cc(c[2], a[2], b[2]);
	subc_cc(c[1], a[1], b[1]);
	subc_cc(c[0], a[0], b[0]);

	unsigned int borrow = 0;
	subc(borrow, 0, 0);

	return (borrow & 0x01);
}


__device__ static void addModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	add_cc(c[7], a[7], b[7]);
	addc_cc(c[6], a[6], b[6]);
	addc_cc(c[5], a[5], b[5]);
	addc_cc(c[4], a[4], b[4]);
	addc_cc(c[3], a[3], b[3]);
	addc_cc(c[2], a[2], b[2]);
	addc_cc(c[1], a[1], b[1]);
	addc_cc(c[0], a[0], b[0]);

	unsigned int carry = 0;
	addc(carry, 0, 0);

	bool gt = false;
	for(int i = 0; i < 8; i++) {
		if(c[i] > _P[i]) {
			gt = true;
			break;
		} else if(c[i] < _P[i]) {
			break;
		}
	}

	if(carry || gt) {
		sub_cc(c[7], c[7], _P[7]);
		subc_cc(c[6], c[6], _P[6]);
		subc_cc(c[5], c[5], _P[5]);
		subc_cc(c[4], c[4], _P[4]);
		subc_cc(c[3], c[3], _P[3]);
		subc_cc(c[2], c[2], _P[2]);
		subc_cc(c[1], c[1], _P[1]);
		subc(c[0], c[0], _P[0]);
	}
}



__device__ static void mulModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
	unsigned int high[8] = { 0 };

	unsigned int t = a[7];

	// a[7] * b (low)
	for(int i = 7; i >= 0; i--) {
		c[i] = t * b[i];
	}

	// a[7] * b (high)
	mad_hi_cc(c[6], t, b[7], c[6]);
	madc_hi_cc(c[5], t, b[6], c[5]);
	madc_hi_cc(c[4], t, b[5], c[4]);
	madc_hi_cc(c[3], t, b[4], c[3]);
	madc_hi_cc(c[2], t, b[3], c[2]);
	madc_hi_cc(c[1], t, b[2], c[1]);
	madc_hi_cc(c[0], t, b[1], c[0]);
	madc_hi(high[7], t, b[0], high[7]);



	// a[6] * b (low)
	t = a[6];
	mad_lo_cc(c[6], t, b[7], c[6]);
	madc_lo_cc(c[5], t, b[6], c[5]);
	madc_lo_cc(c[4], t, b[5], c[4]);
	madc_lo_cc(c[3], t, b[4], c[3]);
	madc_lo_cc(c[2], t, b[3], c[2]);
	madc_lo_cc(c[1], t, b[2], c[1]);
	madc_lo_cc(c[0], t, b[1], c[0]);
	madc_lo_cc(high[7], t, b[0], high[7]);
	addc(high[6], high[6], 0);

	// a[6] * b (high)
	mad_hi_cc(c[5], t, b[7], c[5]);
	madc_hi_cc(c[4], t, b[6], c[4]);
	madc_hi_cc(c[3], t, b[5], c[3]);
	madc_hi_cc(c[2], t, b[4], c[2]);
	madc_hi_cc(c[1], t, b[3], c[1]);
	madc_hi_cc(c[0], t, b[2], c[0]);
	madc_hi_cc(high[7], t, b[1], high[7]);
	madc_hi(high[6], t, b[0], high[6]);

	// a[5] * b (low)
	t = a[5];
	mad_lo_cc(c[5], t, b[7], c[5]);
	madc_lo_cc(c[4], t, b[6], c[4]);
	madc_lo_cc(c[3], t, b[5], c[3]);
	madc_lo_cc(c[2], t, b[4], c[2]);
	madc_lo_cc(c[1], t, b[3], c[1]);
	madc_lo_cc(c[0], t, b[2], c[0]);
	madc_lo_cc(high[7], t, b[1], high[7]);
	madc_lo_cc(high[6], t, b[0], high[6]);
	addc(high[5], high[5], 0);

	// a[5] * b (high)
	mad_hi_cc(c[4], t, b[7], c[4]);
	madc_hi_cc(c[3], t, b[6], c[3]);
	madc_hi_cc(c[2], t, b[5], c[2]);
	madc_hi_cc(c[1], t, b[4], c[1]);
	madc_hi_cc(c[0], t, b[3], c[0]);
	madc_hi_cc(high[7], t, b[2], high[7]);
	madc_hi_cc(high[6], t, b[1], high[6]);
	madc_hi(high[5], t, b[0], high[5]);



	// a[4] * b (low)
	t = a[4];
	mad_lo_cc(c[4], t, b[7], c[4]);
	madc_lo_cc(c[3], t, b[6], c[3]);
	madc_lo_cc(c[2], t, b[5], c[2]);
	madc_lo_cc(c[1], t, b[4], c[1]);
	madc_lo_cc(c[0], t, b[3], c[0]);
	madc_lo_cc(high[7], t, b[2], high[7]);
	madc_lo_cc(high[6], t, b[1], high[6]);
	madc_lo_cc(high[5], t, b[0], high[5]);
	addc(high[4], high[4], 0);

	// a[4] * b (high)
	mad_hi_cc(c[3], t, b[7], c[3]);
	madc_hi_cc(c[2], t, b[6], c[2]);
	madc_hi_cc(c[1], t, b[5], c[1]);
	madc_hi_cc(c[0], t, b[4], c[0]);
	madc_hi_cc(high[7], t, b[3], high[7]);
	madc_hi_cc(high[6], t, b[2], high[6]);
	madc_hi_cc(high[5], t, b[1], high[5]);
	madc_hi(high[4], t, b[0], high[4]);



	// a[3] * b (low)
	t = a[3];
	mad_lo_cc(c[3], t, b[7], c[3]);
	madc_lo_cc(c[2], t, b[6], c[2]);
	madc_lo_cc(c[1], t, b[5], c[1]);
	madc_lo_cc(c[0], t, b[4], c[0]);
	madc_lo_cc(high[7], t, b[3], high[7]);
	madc_lo_cc(high[6], t, b[2], high[6]);
	madc_lo_cc(high[5], t, b[1], high[5]);
	madc_lo_cc(high[4], t, b[0], high[4]);
	addc(high[3], high[3], 0);

	// a[3] * b (high)
	mad_hi_cc(c[2], t, b[7], c[2]);
	madc_hi_cc(c[1], t, b[6], c[1]);
	madc_hi_cc(c[0], t, b[5], c[0]);
	madc_hi_cc(high[7], t, b[4], high[7]);
	madc_hi_cc(high[6], t, b[3], high[6]);
	madc_hi_cc(high[5], t, b[2], high[5]);
	madc_hi_cc(high[4], t, b[1], high[4]);
	madc_hi(high[3], t, b[0], high[3]);



	// a[2] * b (low)
	t = a[2];
	mad_lo_cc(c[2], t, b[7], c[2]);
	madc_lo_cc(c[1], t, b[6], c[1]);
	madc_lo_cc(c[0], t, b[5], c[0]);
	madc_lo_cc(high[7], t, b[4], high[7]);
	madc_lo_cc(high[6], t, b[3], high[6]);
	madc_lo_cc(high[5], t, b[2], high[5]);
	madc_lo_cc(high[4], t, b[1], high[4]);
	madc_lo_cc(high[3], t, b[0], high[3]);
	addc(high[2], high[2], 0);

	// a[2] * b (high)
	mad_hi_cc(c[1], t, b[7], c[1]);
	madc_hi_cc(c[0], t, b[6], c[0]);
	madc_hi_cc(high[7], t, b[5], high[7]);
	madc_hi_cc(high[6], t, b[4], high[6]);
	madc_hi_cc(high[5], t, b[3], high[5]);
	madc_hi_cc(high[4], t, b[2], high[4]);
	madc_hi_cc(high[3], t, b[1], high[3]);
	madc_hi(high[2], t, b[0], high[2]);



	// a[1] * b (low)
	t = a[1];
	mad_lo_cc(c[1], t, b[7], c[1]);
	madc_lo_cc(c[0], t, b[6], c[0]);
	madc_lo_cc(high[7], t, b[5], high[7]);
	madc_lo_cc(high[6], t, b[4], high[6]);
	madc_lo_cc(high[5], t, b[3], high[5]);
	madc_lo_cc(high[4], t, b[2], high[4]);
	madc_lo_cc(high[3], t, b[1], high[3]);
	madc_lo_cc(high[2], t, b[0], high[2]);
	addc(high[1], high[1], 0);

	// a[1] * b (high)
	mad_hi_cc(c[0], t, b[7], c[0]);
	madc_hi_cc(high[7], t, b[6], high[7]);
	madc_hi_cc(high[6], t, b[5], high[6]);
	madc_hi_cc(high[5], t, b[4], high[5]);
	madc_hi_cc(high[4], t, b[3], high[4]);
	madc_hi_cc(high[3], t, b[2], high[3]);
	madc_hi_cc(high[2], t, b[1], high[2]);
	madc_hi(high[1], t, b[0], high[1]);



	// a[0] * b (low)
	t = a[0];
	mad_lo_cc(c[0], t, b[7], c[0]);
	madc_lo_cc(high[7], t, b[6], high[7]);
	madc_lo_cc(high[6], t, b[5], high[6]);
	madc_lo_cc(high[5], t, b[4], high[5]);
	madc_lo_cc(high[4], t, b[3], high[4]);
	madc_lo_cc(high[3], t, b[2], high[3]);
	madc_lo_cc(high[2], t, b[1], high[2]);
	madc_lo_cc(high[1], t, b[0], high[1]);
	addc(high[0], high[0], 0);

	// a[0] * b (high)
	mad_hi_cc(high[7], t, b[7], high[7]);
	madc_hi_cc(high[6], t, b[6], high[6]);
	madc_hi_cc(high[5], t, b[5], high[5]);
	madc_hi_cc(high[4], t, b[4], high[4]);
	madc_hi_cc(high[3], t, b[3], high[3]);
	madc_hi_cc(high[2], t, b[2], high[2]);
	madc_hi_cc(high[1], t, b[1], high[1]);
	madc_hi(high[0], t, b[0], high[0]);



	// At this point we have 16 32-bit words representing a 512-bit value
	// high[0 ... 7] and c[0 ... 7]
	const unsigned int s = 977;

	// Store high[6] and high[7] since they will be overwritten
	unsigned int high7 = high[7];
	unsigned int high6 = high[6];


	// Take high 256 bits, multiply by 2^32, add to low 256 bits
	// That is, take high[0 ... 7], shift it left 1 word and add it to c[0 ... 7]
	add_cc(c[6], high[7], c[6]);
	addc_cc(c[5], high[6], c[5]);
	addc_cc(c[4], high[5], c[4]);
	addc_cc(c[3], high[4], c[3]);
	addc_cc(c[2], high[3], c[2]);
	addc_cc(c[1], high[2], c[1]);
	addc_cc(c[0], high[1], c[0]);
	addc_cc(high[7], high[0], 0);
	addc(high[6], 0, 0);


	// Take high 256 bits, multiply by 977, add to low 256 bits
	// That is, take high[0 ... 5], high6, high7, multiply by 977 and add to c[0 ... 7]
	mad_lo_cc(c[7], high7, s, c[7]);
	madc_lo_cc(c[6], high6, s, c[6]);
	madc_lo_cc(c[5], high[5], s, c[5]);
	madc_lo_cc(c[4], high[4], s, c[4]);
	madc_lo_cc(c[3], high[3], s, c[3]);
	madc_lo_cc(c[2], high[2], s, c[2]);
	madc_lo_cc(c[1], high[1], s, c[1]);
	madc_lo_cc(c[0], high[0], s, c[0]);
	addc_cc(high[7], high[7], 0);
	addc(high[6], high[6], 0);


	mad_hi_cc(c[6], high7, s, c[6]);
	madc_hi_cc(c[5], high6, s, c[5]);
	madc_hi_cc(c[4], high[5], s, c[4]);
	madc_hi_cc(c[3], high[4], s, c[3]);
	madc_hi_cc(c[2], high[3], s, c[2]);
	madc_hi_cc(c[1], high[2], s, c[1]);
	madc_hi_cc(c[0], high[1], s, c[0]);
	madc_hi_cc(high[7], high[0], s, high[7]);
	addc(high[6], high[6], 0);


	// Repeat the same steps, but this time we only need to handle high[6] and high[7]
	high7 = high[7];
	high6 = high[6];

	// Take the high 64 bits, multiply by 2^32 and add to the low 256 bits
	add_cc(c[6], high[7], c[6]);
	addc_cc(c[5], high[6], c[5]);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], 0, 0);


	// Take the high 64 bits, multiply by 977 and add to the low 256 bits
	mad_lo_cc(c[7], high7, s, c[7]);
	madc_lo_cc(c[6], high6, s, c[6]);
	addc_cc(c[5], c[5], 0);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], high[7], 0);

	mad_hi_cc(c[6], high7, s, c[6]);
	madc_hi_cc(c[5], high6, s, c[5]);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], high[7], 0);


	bool overflow = high[7] != 0;

	unsigned int borrow = sub(c, _P, c);

	if(overflow) {
		if(!borrow) {
			sub(c, _P, c);
		}
	} else {
		if(borrow) {
			add(c, _P, c);
		}
	}
}


/**
 * Square mod P
 * b = a * a
 */
__device__ static void squareModP(const unsigned int a[8], unsigned int b[8])
{
	mulModP(a, a, b);
}

/**
 * Square mod P
 * x = x * x
 */
__device__ static void squareModP(unsigned int x[8])
{
	unsigned int tmp[8];
	squareModP(x, tmp);
	copyBigInt(tmp, x);
}

/**
 * Multiply mod P
 * c = a * c
 */
__device__ static void mulModP(const unsigned int a[8], unsigned int c[8])
{
	unsigned int tmp[8];
	mulModP(a, c, tmp);

	copyBigInt(tmp, c);
}

/**
 * Multiplicative inverse mod P using Fermat's method of x^(p-2) mod p and addition chains
 */
__device__ static void invModP(unsigned int value[8])
{
	unsigned int x[8];

	copyBigInt(value, x);

	unsigned int y[8] = { 0, 0, 0, 0, 0, 0, 0, 1 };

	// 0xd - 1101
	mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);


	// 0x2 - 0010
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);

	// 0xc = 0x1100
	//mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);

	// 0xfffff
	for(int i = 0; i < 20; i++) {
		mulModP(x, y);
		squareModP(x);
	}

	// 0xe - 1110
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);

	// 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffff
	for(int i = 0; i < 219; i++) {
		mulModP(x, y);
		squareModP(x);
	}
	mulModP(x, y);

	copyBigInt(y, value);
}

__device__ static void invModP(const unsigned int *value, unsigned int *inverse)
{
	copyBigInt(value, inverse);

	invModP(inverse);
}

__device__ static void negModP(const unsigned int *value, unsigned int *negative)
{
	sub_cc(negative[0], _P[0], value[0]);
	subc_cc(negative[1], _P[1], value[1]);
	subc_cc(negative[2], _P[2], value[2]);
	subc_cc(negative[3], _P[3], value[3]);
	subc_cc(negative[4], _P[4], value[4]);
	subc_cc(negative[5], _P[5], value[5]);
	subc_cc(negative[6], _P[6], value[6]);
	subc(negative[7], _P[7], value[7]);
}


__device__ __forceinline__ static void beginBatchAdd(const unsigned int *px, const unsigned int *x, unsigned int *chain, int i, int batchIdx, unsigned int inverse[8])
{
	// x = Gx - x
	unsigned int t[8];
	subModP(px, x, t);

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(t, inverse);

	writeInt(chain, batchIdx, inverse);
}


__device__ __forceinline__ static void beginBatchAddWithDouble(const unsigned int *px, const unsigned int *py, unsigned int *xPtr, unsigned int *chain, int i, int batchIdx, unsigned int inverse[8])
{
	unsigned int x[8];
	readInt(xPtr, i, x);

	if(equal(px, x)) {
		addModP(py, py, x);
	} else {
		// x = Gx - x
		subModP(px, x, x);
	}

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(x, inverse);

	writeInt(chain, batchIdx, inverse);
}

__device__ static void completeBatchAddWithDouble(const unsigned int *px, const unsigned int *py, const unsigned int *xPtr, const unsigned int *yPtr, int i, int batchIdx, unsigned int *chain, unsigned int *inverse, unsigned int newX[8], unsigned int newY[8])
{
	unsigned int s[8];
	unsigned int x[8];
	unsigned int y[8];

	readInt(xPtr, i, x);
	readInt(yPtr, i, y);

	if(batchIdx >= 1) {
		unsigned int c[8];

		readInt(chain, batchIdx - 1, c);

		mulModP(inverse, c, s);

		unsigned int diff[8];
		if(equal(px, x)) {
			addModP(py, py, diff);
		} else {
			subModP(px, x, diff);
		}

		mulModP(diff, inverse);
	} else {
		copyBigInt(inverse, s);
	}


	if(equal(px, x)) {
		// currently s = 1 / 2y

		unsigned int x2[8];
		unsigned int tx2[8];

		// 3x^2
		mulModP(x, x, x2);
		addModP(x2, x2, tx2);
		addModP(x2, tx2, tx2);


		// s = 3x^2 * 1/2y
		mulModP(tx2, s);

		// s^2
		unsigned int s2[8];
		mulModP(s, s, s2);

		// Rx = s^2 - 2px
		subModP(s2, x, newX);
		subModP(newX, x, newX);

		// Ry = s(px - rx) - py
		unsigned int k[8];
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);

	} else {

		unsigned int rise[8];
		subModP(py, y, rise);

		mulModP(rise, s);

		// Rx = s^2 - Gx - Qx
		unsigned int s2[8];
		mulModP(s, s, s2);

		subModP(s2, px, newX);
		subModP(newX, x, newX);

		// Ry = s(px - rx) - py
		unsigned int k[8];
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);
	}
}

__device__ static void completeBatchAdd(const unsigned int *px, const unsigned int *py, unsigned int *xPtr, unsigned int *yPtr, int i, int batchIdx, unsigned int *chain, unsigned int *inverse, unsigned int newX[8], unsigned int newY[8])
{
	unsigned int s[8];
	unsigned int x[8];

	readInt(xPtr, i, x);

	if(batchIdx >= 1) {
		unsigned int c[8];

		readInt(chain, batchIdx - 1, c);
		mulModP(inverse, c, s);

		unsigned int diff[8];
		subModP(px, x, diff);
		mulModP(diff, inverse);
	} else {
		copyBigInt(inverse, s);
	}

	unsigned int y[8];
	readInt(yPtr, i, y);

	unsigned int rise[8];
	subModP(py, y, rise);

	mulModP(rise, s);

	// Rx = s^2 - Gx - Qx
	unsigned int s2[8];
	mulModP(s, s, s2);
	subModP(s2, px, newX);
	subModP(newX, x, newX);

	// Ry = s(px - rx) - py
	unsigned int k[8];
	subModP(px, newX, k);
	mulModP(s, k, newY);
	subModP(newY, py, newY);
}


__device__ __forceinline__ static void doBatchInverse(unsigned int inverse[8])
{
	invModP(inverse);
}

#endif