#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for multinomial
        torch::Tensor weights = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure weights are non-negative (multinomial requires non-negative weights)
        weights = torch::abs(weights);
        
        // Extract parameters for multinomial from remaining data
        int64_t num_samples = 1;
        bool replacement = false;
        
        if (offset + 2 <= Size) {
            // Extract num_samples from the next byte
            num_samples = static_cast<int64_t>(Data[offset++]) % 10 + 1;
            
            // Extract replacement flag from the next byte
            replacement = Data[offset++] & 0x1;
        }
        
        // Try different tensor shapes and edge cases
        if (weights.dim() == 0) {
            // Scalar tensor - reshape to 1D
            weights = weights.reshape({1});
        } else if (weights.dim() == 1) {
            // 1D tensor - already in correct format
        } else {
            // For higher dimensions, flatten to 1D or use the last dimension
            if (weights.dim() > 1) {
                if (Data[offset % Size] % 2 == 0) {
                    // Flatten to 1D
                    weights = weights.flatten();
                } else {
                    // Use the last dimension
                    int64_t last_dim = weights.size(-1);
                    weights = weights.reshape({-1, last_dim});
                }
            }
        }
        
        // Apply multinomial operation
        torch::Tensor result;
        
        // Try different variants of multinomial
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 2;
            
            switch (variant) {
                case 0:
                    // Basic multinomial
                    result = torch::multinomial(weights, num_samples, replacement);
                    break;
                    
                case 1:
                    // Multinomial with generator
                    {
                        auto gen = torch::Generator();
                        if (offset < Size) {
                            gen.set_current_seed(Data[offset++]);
                        }
                        result = torch::multinomial(weights, num_samples, replacement, gen);
                    }
                    break;
            }
        } else {
            // Default to basic multinomial if no more data
            result = torch::multinomial(weights, num_samples, replacement);
        }
        
        // Access result to ensure computation is not optimized away
        auto sum = result.sum().item<int64_t>();
        (void)sum;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}