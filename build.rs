// build.rs

// extern crate bindgen; // Not needed with Rust 2018+ edition, just use `use`

use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to invalidate the built crate whenever the wrapper header changes
    println!("cargo:rerun-if-changed=wrapper_header.h");
    // Also, if bitcrack_capi.h changes (assuming it's one level up)
    println!("cargo:rerun-if-changed=../bitcrack_capi.h");


    // --- Linker configuration ---
    // This part is highly dependent on how the `libbitcrack_shared` C library is built
    // and where its artifacts are located.

    // Option 1: Use an environment variable to specify the library path.
    // User would set BITCRACK_LIB_DIR=/path/to/bitcrack_project/build/lib (or similar)
    if let Ok(lib_dir) = env::var("BITCRACK_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else {
        // Option 2: Assume a common relative path if the Rust crate is part of the BitCrack build system.
        // This might point to where CMake builds the shared library.
        // Adjust this relative path as needed. For example, if Rust crate is in `bindings/rust`
        // and the library is in `../../build/lib` relative to the crate's root.
        // For now, this is a placeholder that will likely need user adjustment.
        println!("cargo:rustc-link-search=native=../../build"); // Example relative path
        println!("cargo:rustc-link-search=native=../../build/lib"); // Common variation
        println!("cargo:rustc-link-search=native=../target/debug"); // If bitcrack_shared is also a rust crate (not the case here)
        println!("cargo:rustc-link-search=native=/usr/local/lib"); // Standard install path
    }

    // Link to the dylib/shared object itself
    // On Linux: libbitcrack_shared.so -> cargo:rustc-link-lib=dylib=bitcrack_shared
    // On macOS: libbitcrack_shared.dylib -> cargo:rustc-link-lib=dylib=bitcrack_shared
    // On Windows: bitcrack_shared.dll (linked via .lib) -> cargo:rustc-link-lib=dylib=bitcrack_shared (usually)
    println!("cargo:rustc-link-lib=dylib=bitcrack_shared");


    // --- Bindgen setup ---
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("wrapper_header.h")
        // Add include path for bindgen to find "bitcrack_capi.h" via wrapper_header.h
        // Assuming bitcrack_capi.h is in the parent directory of this Rust crate's root.
        // And also the binary dir for bitcrack_export.h
        // These paths might need to be configured via env vars for more robustness.
        .clang_arg("-I../") // To find bitcrack_capi.h if it's in parent dir
        .clang_arg(format!("-I{}/include", env::var("BITCRACK_BUILD_DIR").unwrap_or_else(|_| "../../build".to_string()))) // To find bitcrack_export.h

        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))

        // Whitelist functions and types from the C API to avoid pulling in too much from std headers.
        .allowlist_function("bitcrack_.*")
        .allowlist_type("BitCrack.*") // Catches BitCrackSession and BitCrackFoundKeyC
        .allowlist_type("PFN_BitCrackResultCallback")

        // Make the BitCrackSessionOpaque an opaque type in Rust, as its definition is private in C API.
        .opaque_type("BitCrackSessionOpaque")

        // Invalidate the build cache if any of the included headers change
        // (handled by parse_callbacks with CargoCallbacks::new())

        // Generate the bindings.
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
