// This header is used by bindgen to generate Rust FFI bindings.
// It should include the main C API header for BitCrack.

// Adjust this path based on the actual location of bitcrack_capi.h
// relative to where build.rs is run, or ensure bitcrack_capi.h
// is in an include path accessible to bindgen.
// If this Rust project is a subdirectory like `bindings/rust/bitcrack_rs`,
// and bitcrack_capi.h is at the BitCrack project root, this might be:
// #include "../../bitcrack_capi.h"
// For now, assuming it can be found directly or via include paths set for bindgen.
#include "bitcrack_capi.h"
