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
        
        // Need at least 1 byte for window length
        if (Size < 1) {
            return 0;
        }
        
        // Parse window length from the first byte
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            window_length = static_cast<int64_t>(Data[offset++]);
        }
        
        // Parse beta parameter if we have more data
        double beta = 12.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse periodic flag if we have more data
        bool periodic = false;
        if (offset < Size) {
            periodic = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Parse dtype if we have more data
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Parse layout if we have more data
        torch::Layout layout = torch::kStrided;
        if (offset < Size && (Data[offset] & 0x01)) {
            layout = torch::kSparse;
        }
        if (offset < Size) {
            offset++;
        }
        
        // Parse device if we have more data
        torch::Device device = torch::kCPU;
        if (offset < Size) {
            int device_index = static_cast<int>(Data[offset++]) % 8; // Limit device index
            device = torch::Device(torch::kCPU, device_index);
        }
        
        // Create options
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .layout(layout)
            .device(device);
        
        // Call kaiser_window with various parameters
        try {
            auto window = torch::kaiser_window(window_length, periodic, options);
        } catch (const std::exception& e) {
            // Expected exceptions for invalid inputs are fine
        }
        
        // Try with explicit beta parameter
        try {
            auto window = torch::kaiser_window(window_length, periodic, beta, options);
        } catch (const std::exception& e) {
            // Expected exceptions for invalid inputs are fine
        }
        
        // Try with different window lengths
        if (offset + 1 < Size) {
            int64_t alt_length = static_cast<int64_t>(Data[offset++]);
            try {
                auto window = torch::kaiser_window(alt_length, !periodic, beta + 1.0, options);
            } catch (const std::exception& e) {
                // Expected exceptions for invalid inputs are fine
            }
        }
        
        // Try with extreme beta values
        if (offset + sizeof(double) <= Size) {
            double extreme_beta;
            std::memcpy(&extreme_beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            try {
                auto window = torch::kaiser_window(window_length, periodic, extreme_beta, options);
            } catch (const std::exception& e) {
                // Expected exceptions for invalid inputs are fine
            }
        }
        
        // Try with different dtypes
        if (offset < Size) {
            auto alt_dtype = fuzzer_utils::parseDataType(Data[offset++]);
            auto alt_options = options.dtype(alt_dtype);
            try {
                auto window = torch::kaiser_window(window_length, periodic, beta, alt_options);
            } catch (const std::exception& e) {
                // Expected exceptions for invalid inputs are fine
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
