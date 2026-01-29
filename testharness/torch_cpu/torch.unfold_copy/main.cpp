#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least a few bytes for tensor creation and unfold parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip if tensor is empty or scalar
        if (input.dim() == 0 || input.numel() == 0) {
            return 0;
        }
        
        // Extract parameters for unfold operation
        // We need at least 3 bytes for dimension, size, and step
        if (offset + 3 > Size) {
            return 0;
        }
        
        // Get dimension parameter - constrain to valid range
        int64_t dimension = static_cast<int8_t>(Data[offset++]) % input.dim();
        
        // Get size parameter (must be positive and non-zero)
        uint8_t size_param = Data[offset++];
        int64_t size = static_cast<int64_t>((size_param % 16) + 1); // 1 to 16
        
        // Get step parameter (must be positive and non-zero)
        uint8_t step_param = Data[offset++];
        int64_t step = static_cast<int64_t>((step_param % 8) + 1); // 1 to 8
        
        // Ensure size doesn't exceed the dimension size
        int64_t dim_size = input.size(dimension);
        if (size > dim_size) {
            size = dim_size > 0 ? dim_size : 1;
        }
        
        // Apply unfold_copy operation
        // This may throw for invalid parameter combinations
        try {
            torch::Tensor result = torch::unfold_copy(input, dimension, size, step);
            
            // Optional: perform some basic validation on the result
            if (result.numel() > 0) {
                // Access some elements to ensure the tensor is valid
                auto flat = result.flatten();
                if (flat.numel() > 0) {
                    flat[0].item<double>(); // Use double to handle more dtypes
                }
            }
        } catch (const c10::Error &) {
            // Expected failures for invalid parameter combinations - silently ignore
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
}