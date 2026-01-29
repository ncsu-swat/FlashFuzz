#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For memcpy

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip if tensor has no dimensions
        if (input.dim() == 0) {
            return 0;
        }
        
        // Need at least 1 byte for number of dimensions to pad
        if (offset + 1 >= Size) {
            return 0;
        }
        
        // Get number of dimensions to pad (between 1 and input.dim())
        int64_t max_dims = input.dim();
        uint8_t num_dims_to_pad = (Data[offset++] % max_dims) + 1;
        
        // Need 2 bytes per dimension (for padding before and after)
        if (offset + 2 * num_dims_to_pad + 1 >= Size) {
            return 0;
        }
        
        // Create padding vector
        // constant_pad_nd expects padding in format: [left, right, top, bottom, front, back, ...]
        // for the last num_dims_to_pad dimensions
        std::vector<int64_t> pad;
        pad.reserve(2 * num_dims_to_pad);
        
        // Fill padding values - use unsigned values to avoid negative padding issues
        for (int i = 0; i < num_dims_to_pad; i++) {
            // Get padding before and after for this dimension
            // Use modulo to keep padding reasonable (0-31 range)
            int64_t pad_before = Data[offset++] % 32;
            int64_t pad_after = Data[offset++] % 32;
            
            // Padding is specified from last dimension to first
            pad.push_back(pad_before);
            pad.push_back(pad_after);
        }
        
        // Get value to pad with
        double pad_value = 0.0;
        if (offset + 1 <= Size) {
            // Use a byte to create a simple pad value
            pad_value = static_cast<double>(static_cast<int8_t>(Data[offset++]));
        }
        
        // Inner try-catch for expected failures (shape mismatches, etc.)
        try {
            // Apply constant_pad_nd operation
            torch::Tensor output = torch::constant_pad_nd(input, pad, pad_value);
            
            // Ensure the output is used to prevent optimization
            if (output.numel() > 0) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        }
        catch (const c10::Error &e) {
            // Expected errors from invalid combinations - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}