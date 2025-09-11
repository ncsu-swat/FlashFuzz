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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to use for max operation (if needed)
        int64_t dim = 0;
        bool keepdim = false;
        
        // If we have more data, use it to determine dimension and keepdim
        if (offset + 2 < Size) {
            // Extract a dimension value from the data
            dim = static_cast<int64_t>(Data[offset++]) % (input.dim() + 1) - 1; // -1 means no dimension specified
            keepdim = Data[offset++] & 0x1; // Use lowest bit to determine keepdim
        }
        
        // Test different variants of torch::max
        
        // Variant 1: max of all elements
        torch::Tensor result1 = torch::max(input);
        
        // Variant 2: max along dimension
        if (input.dim() > 0) {
            if (dim >= 0) {
                // Max along specific dimension
                auto result2 = torch::max(input, dim, keepdim);
                torch::Tensor values = std::get<0>(result2);
                torch::Tensor indices = std::get<1>(result2);
            }
        }
        
        // Variant 3: element-wise maximum of two tensors
        if (offset < Size) {
            // Create a second tensor if we have more data
            torch::Tensor other;
            try {
                other = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try element-wise max if shapes are broadcastable
                try {
                    torch::Tensor result3 = torch::max(input, other);
                } catch (const c10::Error &) {
                    // Shapes might not be broadcastable, which is expected in some cases
                }
            } catch (const std::exception &) {
                // Failed to create second tensor, continue with other tests
            }
        }
        
        // Variant 4: max with scalar
        if (offset < Size) {
            // Use a byte from the input as a scalar value
            double scalar_value = static_cast<double>(Data[offset++]);
            auto result4_tuple = torch::max(input, scalar_value);
            torch::Tensor result4 = std::get<0>(result4_tuple);
        }
        
        // Variant 5: named max along dimension
        if (input.dim() > 0 && dim >= 0) {
            try {
                auto result5 = torch::max(input, torch::Dimname::wildcard(), keepdim);
            } catch (const c10::Error &) {
                // Named dimensions might not be supported for this tensor
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
