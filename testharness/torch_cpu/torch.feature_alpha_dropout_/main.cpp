#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least a few bytes for tensor creation and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor - feature_alpha_dropout expects float tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if needed (alpha dropout requires float types)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Feature alpha dropout requires at least 2D input
        // Reshape if necessary
        if (input.dim() < 2) {
            auto numel = input.numel();
            if (numel >= 2) {
                input = input.view({1, -1});
            } else {
                // Not enough elements, create a minimal valid tensor
                input = torch::randn({1, 4});
            }
        }
        
        // Make tensor contiguous for inplace operation
        input = input.contiguous();
        
        // Extract p (dropout probability) from the input data
        double p = 0.5;
        if (offset + sizeof(float) <= Size) {
            float p_float;
            std::memcpy(&p_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1)
            p_float = std::abs(p_float);
            p_float = p_float - std::floor(p_float);
            // Ensure p < 1.0 to avoid issues
            if (p_float >= 1.0f) {
                p_float = 0.5f;
            }
            p = static_cast<double>(p_float);
        }
        
        // Extract training flag from the input data
        bool training = true;
        if (offset < Size) {
            training = (Data[offset++] & 0x1) != 0;
        }
        
        // Store original shape for verification
        auto original_sizes = input.sizes().vec();
        
        // Clone input for testing both inplace and non-inplace versions
        torch::Tensor input_clone = input.clone();
        
        // Test the inplace version (torch.feature_alpha_dropout_)
        try {
            torch::feature_alpha_dropout_(input, p, training);
            
            // Verify shape is preserved
            if (input.sizes().vec() != original_sizes) {
                // Shape mismatch - this would be a bug
            }
        } catch (const c10::Error&) {
            // Expected for some invalid configurations
        }
        
        // Also test non-inplace version for comparison
        try {
            torch::Tensor output = torch::feature_alpha_dropout(input_clone, p, training);
            
            // Verify output is defined and has correct shape
            if (output.defined() && output.sizes().vec() != original_sizes) {
                // Shape mismatch - this would be a bug
            }
        } catch (const c10::Error&) {
            // Expected for some invalid configurations
        }
        
        // Test edge cases based on fuzzer data
        if (offset < Size) {
            uint8_t test_case = Data[offset++] % 4;
            try {
                switch (test_case) {
                    case 0: {
                        // Test with p=0 (no dropout)
                        torch::Tensor temp = input_clone.clone();
                        torch::feature_alpha_dropout_(temp, 0.0, training);
                        break;
                    }
                    case 1: {
                        // Test with training=false (should be identity)
                        torch::Tensor temp = input_clone.clone();
                        torch::feature_alpha_dropout_(temp, p, false);
                        break;
                    }
                    case 2: {
                        // Test with higher dimensional input
                        if (input_clone.dim() == 2) {
                            torch::Tensor reshaped = input_clone.unsqueeze(0);
                            torch::feature_alpha_dropout_(reshaped, p, training);
                        }
                        break;
                    }
                    case 3: {
                        // Test with different p values
                        torch::Tensor temp1 = input_clone.clone();
                        torch::Tensor temp2 = input_clone.clone();
                        torch::feature_alpha_dropout_(temp1, 0.1, training);
                        torch::feature_alpha_dropout_(temp2, 0.9, training);
                        break;
                    }
                }
            } catch (const c10::Error&) {
                // Expected for some configurations
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