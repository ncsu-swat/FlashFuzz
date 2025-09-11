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
        
        // Extract window_length from the input data
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            // Not enough data for window_length, use a default value
            window_length = Data[offset++];
        }
        
        // Extract periodic flag (true/false)
        bool periodic = false;
        if (offset < Size) {
            periodic = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Extract dtype
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Extract layout
        torch::Layout layout = torch::kStrided;
        if (offset < Size && (Data[offset] & 0x01)) {
            layout = torch::kSparse;
        }
        if (offset < Size) {
            offset++;
        }
        
        // Extract device
        torch::Device device = torch::kCPU;
        
        // Create options
        auto options = torch::TensorOptions()
                           .dtype(dtype)
                           .layout(layout)
                           .device(device);
        
        // Call bartlett_window with different parameters
        try {
            auto window = torch::bartlett_window(window_length, options);
        } catch (...) {
            // Ignore exceptions from the operation itself
        }
        
        try {
            auto window_periodic = torch::bartlett_window(window_length, periodic, options);
        } catch (...) {
            // Ignore exceptions from the operation itself
        }
        
        // Try with a tensor input for window_length
        if (offset < Size) {
            try {
                auto tensor_input = fuzzer_utils::createTensor(Data, Size, offset);
                if (tensor_input.dim() == 0 && tensor_input.scalar_type() == torch::kLong) {
                    int64_t tensor_value = tensor_input.item<int64_t>();
                    auto window = torch::bartlett_window(tensor_value, options);
                }
            } catch (...) {
                // Ignore exceptions from tensor creation or the operation
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
