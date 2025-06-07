#ifndef BITCRACK_CAPI_H
#define BITCRACK_CAPI_H

// This path assumes bitcrack_export.h will be in the include path during compilation
// e.g. from CMAKE_CURRENT_BINARY_DIR/include if bitcrack_capi.h is in a root include dir
// or just "bitcrack_export.h" if both are in the same include destination.
// For the target_include_directories in CMake, ${CMAKE_CURRENT_BINARY_DIR} itself (or its 'include' subdir)
// should be an include path for the library build.
#include "bitcrack_export.h"
#include <stdint.h>
#include <stddef.h> // For size_t

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to the BitCrack session/context
typedef struct BitCrackSessionOpaque* BitCrackSession;

// Basic result structure
typedef struct {
    char private_key_hex[65]; // 64 chars + null terminator
    char public_key_hex[131]; // Max uncompressed "04" + X + Y + null
    char address_base58[60];  // Max Base58 address length (e.g., P2PKH) + null
    int is_compressed;        // 0 for false, 1 for true
} BitCrackFoundKeyC;

// Callback function type for when a key is found
typedef void (*PFN_BitCrackResultCallback)(const BitCrackFoundKeyC* result, void* user_data);

// --- Session Management ---
BITCRACK_API BitCrackSession bitcrack_create_session();
BITCRACK_API void bitcrack_destroy_session(BitCrackSession session);

// --- Configuration ---
// Sets a single GPU device to use.
BITCRACK_API int bitcrack_set_device(BitCrackSession session, int device_id, int blocks, int threads, int points_per_thread);
BITCRACK_API int bitcrack_set_keyspace(BitCrackSession session, const char* start_key_hex, const char* end_key_hex);
// Sets target addresses from a list of strings.
BITCRACK_API int bitcrack_add_targets(BitCrackSession session, const char* const* addresses, int num_addresses);
// Sets compression mode: 0=COMPRESSED, 1=UNCOMPRESSED, 2=BOTH
BITCRACK_API int bitcrack_set_compression_mode(BitCrackSession session, int mode);

// --- Execution & Results ---
BITCRACK_API int bitcrack_init_search(BitCrackSession session); // Initializes device, targets, etc.
BITCRACK_API int bitcrack_start_search_async(BitCrackSession session); // Starts search in a new thread managed by the lib
BITCRACK_API void bitcrack_stop_search(BitCrackSession session);     // Signals the search thread to stop
BITCRACK_API int bitcrack_is_search_running(BitCrackSession session); // Returns 1 if running, 0 otherwise

BITCRACK_API void bitcrack_set_result_callback(BitCrackSession session, PFN_BitCrackResultCallback callback, void* user_data);

// Polling for results
BITCRACK_API int bitcrack_poll_results(BitCrackSession session, BitCrackFoundKeyC* out_results, int max_results_to_fetch, int* num_results_fetched);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BITCRACK_CAPI_H
