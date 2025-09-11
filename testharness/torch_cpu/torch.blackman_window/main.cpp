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
        
        // Parse window length from input data
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            // If not enough data, use a single byte
            window_length = static_cast<int64_t>(Data[offset++]);
        }
        
        // Parse periodic flag (if we have data left)
        bool periodic = false;
        if (offset < Size) {
            periodic = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Parse layout (if we have data left)
        torch::Layout layout = torch::kStrided;
        if (offset < Size) {
            uint8_t layout_byte = Data[offset++];
            if (layout_byte % 2 == 1) {
                layout = torch::kSparse;
            }
        }
        
        // Parse device (if we have data left)
        torch::Device device = torch::kCPU;
        if (offset < Size) {
            // We'll just use CPU for now, but could add GPU support
            // uint8_t device_byte = Data[offset++];
            offset++;
        }
        
        // Parse dtype (if we have data left)
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Create options
        auto options = torch::TensorOptions()
            .layout(layout)
            .device(device)
            .dtype(dtype);
        
        // Call blackman_window with different combinations of parameters
        torch::Tensor result;
        
        // Try different variants of the function
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 4;
            
            switch (variant) {
                case 0:
                    // Basic call with just window_length
                    result = torch::blackman_window(window_length);
                    break;
                    
                case 1:
                    // Call with window_length and periodic flag
                    result = torch::blackman_window(window_length, periodic);
                    break;
                    
                case 2:
                    // Call with window_length, periodic flag, and options
                    result = torch::blackman_window(window_length, periodic, options);
                    break;
                    
                case 3:
                    // Call with window_length and options (no periodic flag)
                    result = torch::blackman_window(window_length, options);
                    break;
            }
        } else {
            // Default to basic call if no variant byte
            result = torch::blackman_window(window_length);
        }
        
        // Perform some operations on the result to ensure it's used
        if (result.defined()) {
            auto sum = result.sum();
            auto max_val = result.max();
            auto min_val = result.min();
            
            // Force evaluation
            sum.item<double>();
            max_val.item<double>();
            min_val.item<double>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
