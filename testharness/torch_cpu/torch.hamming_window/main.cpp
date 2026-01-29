#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For isnan, isinf

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for window_length
        if (Size < 1) {
            return 0;
        }
        
        // Parse window_length from fuzzer data
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            window_length = static_cast<int64_t>(Data[offset++]);
        }
        
        // Constrain window_length to reasonable range [0, 10000] to avoid OOM
        // and ensure it's non-negative (negative values always throw)
        window_length = std::abs(window_length) % 10001;
        
        // Parse periodic flag (if available)
        bool periodic = false;
        if (offset < Size) {
            periodic = static_cast<bool>(Data[offset++] & 0x1);
        }
        
        // Parse alpha parameter (if available)
        double alpha = 0.54;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize alpha to avoid NaN/Inf
            if (std::isnan(alpha) || std::isinf(alpha)) {
                alpha = 0.54;
            }
            // Clamp to reasonable range
            alpha = std::fmod(std::abs(alpha), 10.0);
        }
        
        // Parse beta parameter (if available)
        double beta = 0.46;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize beta to avoid NaN/Inf
            if (std::isnan(beta) || std::isinf(beta)) {
                beta = 0.46;
            }
            // Clamp to reasonable range
            beta = std::fmod(std::abs(beta), 10.0);
        }
        
        // Parse dtype (if available) - only use float types for window functions
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++] % 4;
            switch (dtype_byte) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kHalf; break;
                case 3: dtype = torch::kBFloat16; break;
            }
        }
        
        // Create options (CPU only for this fuzzer)
        auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
        
        // Call hamming_window with different parameter combinations
        // Use inner try-catch for expected parameter validation failures
        torch::Tensor result;
        
        // Basic call with just window_length
        try {
            result = torch::hamming_window(window_length);
        } catch (...) {}
        
        // Call with periodic flag
        try {
            result = torch::hamming_window(window_length, periodic);
        } catch (...) {}
        
        // Call with periodic and alpha
        try {
            result = torch::hamming_window(window_length, periodic, alpha);
        } catch (...) {}
        
        // Call with all parameters
        try {
            result = torch::hamming_window(window_length, periodic, alpha, beta);
        } catch (...) {}
        
        // Call with options
        try {
            result = torch::hamming_window(window_length, options);
        } catch (...) {}
        
        // Call with options and periodic
        try {
            result = torch::hamming_window(window_length, periodic, options);
        } catch (...) {}
        
        // Call with options, periodic, and alpha
        try {
            result = torch::hamming_window(window_length, periodic, alpha, options);
        } catch (...) {}
        
        // Call with all parameters and options
        try {
            result = torch::hamming_window(window_length, periodic, alpha, beta, options);
        } catch (...) {}
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto sum = result.sum().item<double>();
            (void)sum; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}