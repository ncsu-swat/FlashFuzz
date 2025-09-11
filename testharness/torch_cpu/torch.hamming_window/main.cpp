#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for window_length
        if (Size < 1) {
            return 0;
        }
        
        // Parse window_length from the first byte
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            window_length = static_cast<int64_t>(Data[offset++]);
        }
        
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
        }
        
        // Parse beta parameter (if available)
        double beta = 0.46;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse device type (if available)
        torch::Device device(torch::kCPU);
        if (offset < Size) {
            // Use the byte to determine if we should use CUDA (if available)
            bool use_cuda = (Data[offset++] & 0x1) && torch::cuda::is_available();
            if (use_cuda) {
                device = torch::Device(torch::kCUDA);
            }
        }
        
        // Parse dtype (if available)
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Create options
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        
        // Call hamming_window with different parameter combinations
        torch::Tensor result;
        
        // Basic call with just window_length
        result = torch::hamming_window(window_length);
        
        // Call with periodic flag
        result = torch::hamming_window(window_length, periodic);
        
        // Call with periodic and alpha
        result = torch::hamming_window(window_length, periodic, alpha);
        
        // Call with all parameters
        result = torch::hamming_window(window_length, periodic, alpha, beta);
        
        // Call with options
        result = torch::hamming_window(window_length, options);
        
        // Call with options and periodic
        result = torch::hamming_window(window_length, periodic, options);
        
        // Call with options, periodic, and alpha
        result = torch::hamming_window(window_length, periodic, alpha, options);
        
        // Call with all parameters and options
        result = torch::hamming_window(window_length, periodic, alpha, beta, options);
        
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
