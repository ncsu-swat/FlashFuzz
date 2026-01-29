#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // feature_dropout requires at least 2D tensor (operates on feature dimension)
        // If tensor is 1D or 0D, reshape it to 2D
        if (input.dim() < 2) {
            int64_t numel = input.numel();
            if (numel == 0) {
                return 0; // Can't reshape empty tensor meaningfully
            }
            input = input.view({1, numel});
        }
        
        // Extract probability parameter from the input data
        float p = 0.5f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Handle NaN and Inf
            if (std::isnan(p) || std::isinf(p)) {
                p = 0.5f;
            }
            
            // Ensure p is between 0 and 1
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // Extract train parameter from the input data
        bool train = true; // Default value
        if (offset < Size) {
            train = Data[offset++] & 0x1;
        }
        
        // Apply feature_dropout (non-inplace first to preserve input)
        torch::Tensor output = torch::feature_dropout(input, p, train);
        
        // Verify output has the same shape as input
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output tensor has different shape than input tensor");
        }
        
        // Test with train=false (should be identity)
        torch::Tensor output_eval = torch::feature_dropout(input, p, false);
        
        // Test with train=true
        torch::Tensor output_train = torch::feature_dropout(input, p, true);
        
        // Try with extreme probability values
        if (offset < Size) {
            uint8_t extreme_selector = Data[offset++] % 4;
            float extreme_p;
            
            switch (extreme_selector) {
                case 0: extreme_p = 0.0f; break;      // No dropout
                case 1: extreme_p = 1.0f; break;      // Drop everything
                case 2: extreme_p = 0.999999f; break; // Almost drop everything
                case 3: extreme_p = 0.000001f; break; // Almost no dropout
                default: extreme_p = 0.5f;
            }
            
            torch::Tensor output_extreme = torch::feature_dropout(input, extreme_p, true);
        }
        
        // Test inplace version with a clone
        torch::Tensor input_clone = input.clone();
        torch::Tensor output_inplace = torch::feature_dropout_(input_clone, p, train);
        
        // Try with different tensor shapes if we have enough data
        if (offset + 4 < Size) {
            // Create a batch of feature maps (N, C, ...) format
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure at least 2D
            if (input2.dim() < 2) {
                int64_t numel = input2.numel();
                if (numel > 0) {
                    input2 = input2.view({1, numel});
                    torch::Tensor output2 = torch::feature_dropout(input2, p, train);
                }
            } else {
                torch::Tensor output2 = torch::feature_dropout(input2, p, train);
            }
        }
        
        // Test with 3D tensor (batch, channels, length) - common for 1D convolutions
        if (offset + 2 < Size) {
            uint8_t batch_size = (Data[offset++] % 4) + 1;
            uint8_t channels = (Data[offset++] % 8) + 1;
            int64_t length = (input.numel() > 0) ? input.numel() : 4;
            
            try {
                torch::Tensor input3d = torch::randn({batch_size, channels, length});
                torch::Tensor output3d = torch::feature_dropout(input3d, p, train);
            } catch (...) {
                // Silently ignore shape-related failures
            }
        }
        
        // Test with 4D tensor (batch, channels, height, width) - common for 2D convolutions
        if (offset + 4 < Size) {
            uint8_t batch_size = (Data[offset++] % 4) + 1;
            uint8_t channels = (Data[offset++] % 8) + 1;
            uint8_t height = (Data[offset++] % 8) + 1;
            uint8_t width = (Data[offset++] % 8) + 1;
            
            try {
                torch::Tensor input4d = torch::randn({batch_size, channels, height, width});
                torch::Tensor output4d = torch::feature_dropout(input4d, p, train);
                
                // Also test inplace on 4D
                torch::Tensor input4d_clone = input4d.clone();
                torch::feature_dropout_(input4d_clone, p, train);
            } catch (...) {
                // Silently ignore shape-related failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}