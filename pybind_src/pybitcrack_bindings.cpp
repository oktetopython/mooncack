#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, std::string
#include <vector>
#include <string>
#include <stdexcept> // For throwing Python-catchable exceptions

// Assuming bitcrack_capi.h is in the parent directory (project root)
// relative to pybind_src/
#include "../bitcrack_capi.h"

namespace py = pybind11;

// Helper to check C API return codes and throw Python exceptions
void check_capi_error(int error_code, const std::string& context_message) {
    if (error_code != 0) {
        throw std::runtime_error(context_message + " C API error code: " + std::to_string(error_code));
    }
}

PYBIND11_MODULE(bitcrack_python, m) {
    m.doc() = "Python bindings for BitCrack C API";

    // --- Session Management ---
    m.def("create_session", []() {
        BitCrackSession session = bitcrack_create_session();
        if (!session) {
            throw std::runtime_error("Failed to create BitCrack session (null pointer returned).");
        }
        return reinterpret_cast<uintptr_t>(session);
    }, "Creates a BitCrack session. Returns a session handle (uintptr_t).");

    m.def("destroy_session", [](uintptr_t session_ptr) {
        if (session_ptr == 0) return; // Allow null pointers to be passed without error
        bitcrack_destroy_session(reinterpret_cast<BitCrackSession>(session_ptr));
    }, "Destroys a BitCrack session.", py::arg("session_handle"));

    // --- Configuration ---
    m.def("set_device", [](uintptr_t session_ptr, int device_id, int blocks, int threads, int points_per_thread) {
        if (session_ptr == 0) throw std::invalid_argument("Invalid session handle (null).");
        int ret = bitcrack_set_device(reinterpret_cast<BitCrackSession>(session_ptr), device_id, blocks, threads, points_per_thread);
        check_capi_error(ret, "Failed to set device.");
    }, "Sets GPU device and kernel parameters.",
       py::arg("session_handle"), py::arg("device_id"), py::arg("blocks"), py::arg("threads"), py::arg("points_per_thread"));

    m.def("set_keyspace", [](uintptr_t session_ptr, const std::string& start_key_hex, const std::string& end_key_hex) {
        if (session_ptr == 0) throw std::invalid_argument("Invalid session handle (null).");
        int ret = bitcrack_set_keyspace(reinterpret_cast<BitCrackSession>(session_ptr), start_key_hex.c_str(), end_key_hex.c_str());
        check_capi_error(ret, "Failed to set keyspace.");
    }, "Sets the key search space using hex strings.",
       py::arg("session_handle"), py::arg("start_key_hex"), py::arg("end_key_hex"));

    m.def("add_targets", [](uintptr_t session_ptr, const std::vector<std::string>& addresses) {
        if (session_ptr == 0) throw std::invalid_argument("Invalid session handle (null).");
        std::vector<const char*> c_addresses;
        c_addresses.reserve(addresses.size());
        for (const auto& addr : addresses) {
            c_addresses.push_back(addr.c_str());
        }
        int ret = bitcrack_add_targets(reinterpret_cast<BitCrackSession>(session_ptr), c_addresses.data(), static_cast<int>(c_addresses.size()));
        check_capi_error(ret, "Failed to add targets.");
    }, "Adds target addresses (Base58 strings) from a list.",
       py::arg("session_handle"), py::arg("addresses"));

    m.def("set_compression_mode", [](uintptr_t session_ptr, int mode) {
        if (session_ptr == 0) throw std::invalid_argument("Invalid session handle (null).");
        // Define compression modes in Python module for clarity, e.g., bitcrack_python.COMPRESSED
        // For now, user passes int: 0=COMPRESSED, 1=UNCOMPRESSED, 2=BOTH
        int ret = bitcrack_set_compression_mode(reinterpret_cast<BitCrackSession>(session_ptr), mode);
        check_capi_error(ret, "Failed to set compression mode.");
    }, "Sets point compression mode (0: COMPRESSED, 1: UNCOMPRESSED, 2: BOTH).",
       py::arg("session_handle"), py::arg("mode"));

    // Expose PointCompressionType values to Python module for convenience
    // These values are from CmdParse.h (enum PointCompressionType)
    // COMPRESSED = 0, UNCOMPRESSED = 1, BOTH = 2
    m.attr("COMPRESSION_COMPRESSED") = py::int_(0);
    m.attr("COMPRESSION_UNCOMPRESSED") = py::int_(1);
    m.attr("COMPRESSION_BOTH") = py::int_(2);


    // --- Execution & Results (Initialization part only for this subtask) ---
    m.def("init_search", [](uintptr_t session_ptr) {
        if (session_ptr == 0) throw std::invalid_argument("Invalid session handle (null).");
        int ret = bitcrack_init_search(reinterpret_cast<BitCrackSession>(session_ptr));
        check_capi_error(ret, "Failed to initialize search session.");
    }, "Initializes the search session after configuration.",
       py::arg("session_handle"));

    // Functions related to async execution and callbacks (start_search_async, stop_search, etc.)

    // Expose BitCrackFoundKeyC as a Python class
    py::class_<BitCrackFoundKeyC>(m, "FoundKey")
        .def_property_readonly("private_key_hex", [](const BitCrackFoundKeyC &k) { return std::string(k.private_key_hex); })
        .def_property_readonly("public_key_hex",  [](const BitCrackFoundKeyC &k) { return std::string(k.public_key_hex); })
        .def_property_readonly("address_base58",  [](const BitCrackFoundKeyC &k) { return std::string(k.address_base58); })
        .def_readonly("is_compressed", &BitCrackFoundKeyC::is_compressed)
        .def("__repr__", [](const BitCrackFoundKeyC &k) {
            return "<FoundKey address='" + std::string(k.address_base58) +
                   "' privkey_hex='" + std::string(k.private_key_hex).substr(0,8) + "...'" // Show partial key for brevity
                   " compressed=" + (k.is_compressed ? "True" : "False") + ">";
        });

    m.def("start_search_async", [](uintptr_t session_ptr) {
        if (session_ptr == 0) throw std::invalid_argument("Invalid session handle (null).");
        check_capi_error(bitcrack_start_search_async(reinterpret_cast<BitCrackSession>(session_ptr)), "Failed to start search asynchronously.");
    }, "Starts the key search asynchronously.", py::arg("session_handle"));

    m.def("stop_search", [](uintptr_t session_ptr) {
        if (session_ptr == 0) return; // Allow null pointers
        bitcrack_stop_search(reinterpret_cast<BitCrackSession>(session_ptr));
        // Assuming bitcrack_stop_search itself doesn't return an error code that needs checking here.
        // If it did, we'd use check_capi_error.
    }, "Signals the search to stop and waits for completion.", py::arg("session_handle"));

    m.def("is_search_running", [](uintptr_t session_ptr) {
        if (session_ptr == 0) return false; // Not running if session is null
        return bitcrack_is_search_running(reinterpret_cast<BitCrackSession>(session_ptr)) != 0;
    }, "Checks if the search is currently running.", py::arg("session_handle"));

    m.def("poll_results", [](uintptr_t session_ptr, int max_results_to_fetch = 10) {
        if (session_ptr == 0) throw std::invalid_argument("Invalid session handle (null).");
        if (max_results_to_fetch <= 0) max_results_to_fetch = 10;

        std::vector<BitCrackFoundKeyC> results_buffer(max_results_to_fetch);
        int num_fetched = 0;

        check_capi_error(bitcrack_poll_results(
            reinterpret_cast<BitCrackSession>(session_ptr),
            results_buffer.data(),
            max_results_to_fetch,
            &num_fetched
        ), "Failed to poll results.");

        results_buffer.resize(num_fetched); // Adjust vector to actual number of results fetched
        return results_buffer; // pybind11 converts std::vector<BitCrackFoundKeyC> to a Python list of FoundKey objects
    }, "Polls for found keys. Returns a list of FoundKey objects.",
       py::arg("session_handle"), py::arg("max_results_to_fetch") = 10);

}
