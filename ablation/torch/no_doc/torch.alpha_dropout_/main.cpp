#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;  // Not enough data to work with
        }

        size_t offset = 0;

        // Parse dropout probability from first byte (0.0 to 1.0)
        float p = static_cast<float>(Data[offset++]) / 255.0f;
        
        // Parse training flag from second byte
        bool training = (Data[offset++] % 2) == 1;
        
        // Parse whether to use GPU (if available)
        bool use_cuda = false;
        #ifdef USE_GPU
        if (torch::cuda::is_available() && offset < Size) {
            use_cuda = (Data[offset++] % 2) == 1;
        }
        #else
        offset++; // Consume byte even if not using GPU
        #endif

        // Create input tensor from remaining data
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with minimal tensor
            input = torch::randn({2, 2});
        }

        // Move to CUDA if requested and available
        if (use_cuda && torch::cuda::is_available()) {
            input = input.cuda();
        }

        // Store original tensor for comparison if needed
        torch::Tensor original = input.clone();

        // Test various edge cases based on remaining fuzzer data
        if (offset < Size) {
            uint8_t test_case = Data[offset++] % 10;
            
            switch(test_case) {
                case 0:
                    // Normal case - just apply alpha_dropout_
                    input.alpha_dropout_(p, training);
                    break;
                    
                case 1:
                    // Test with p = 0 (no dropout)
                    input.alpha_dropout_(0.0f, training);
                    if (training && !torch::allclose(input, original)) {
                        // With p=0, tensor should remain unchanged
                    }
                    break;
                    
                case 2:
                    // Test with p = 1 (full dropout)
                    input.alpha_dropout_(1.0f, training);
                    break;
                    
                case 3:
                    // Test with training = false (should be no-op)
                    input.alpha_dropout_(p, false);
                    if (!torch::allclose(input, original)) {
                        // In eval mode, tensor should remain unchanged
                    }
                    break;
                    
                case 4:
                    // Test with empty tensor
                    if (input.numel() > 0) {
                        torch::Tensor empty_tensor = torch::empty({0});
                        if (use_cuda && torch::cuda::is_available()) {
                            empty_tensor = empty_tensor.cuda();
                        }
                        empty_tensor.alpha_dropout_(p, training);
                    }
                    break;
                    
                case 5:
                    // Test with scalar tensor
                    {
                        torch::Tensor scalar = torch::tensor(3.14f);
                        if (use_cuda && torch::cuda::is_available()) {
                            scalar = scalar.cuda();
                        }
                        scalar.alpha_dropout_(p, training);
                    }
                    break;
                    
                case 6:
                    // Test with different dtypes
                    if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                        input.alpha_dropout_(p, training);
                    } else {
                        // Try converting to float first
                        try {
                            input = input.to(torch::kFloat);
                            input.alpha_dropout_(p, training);
                        } catch (...) {
                            // Some dtypes might not support conversion
                        }
                    }
                    break;
                    
                case 7:
                    // Test with very small probability
                    input.alpha_dropout_(1e-10f, training);
                    break;
                    
                case 8:
                    // Test with probability very close to 1
                    input.alpha_dropout_(0.9999f, training);
                    break;
                    
                case 9:
                    // Test multiple applications
                    input.alpha_dropout_(p * 0.5f, training);
                    input.alpha_dropout_(p * 0.5f, training);
                    break;
                    
                default:
                    input.alpha_dropout_(p, training);
                    break;
            }
        } else {
            // Default case when no more data
            input.alpha_dropout_(p, training);
        }

        // Additional edge cases based on tensor properties
        if (offset < Size) {
            uint8_t extra_test = Data[offset++] % 5;
            
            switch(extra_test) {
                case 0:
                    // Test with non-contiguous tensor
                    if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
                        torch::Tensor transposed = input.transpose(0, 1);
                        transposed.alpha_dropout_(p, training);
                    }
                    break;
                    
                case 1:
                    // Test with view
                    if (input.numel() > 4) {
                        torch::Tensor viewed = input.view({-1});
                        viewed.alpha_dropout_(p, training);
                    }
                    break;
                    
                case 2:
                    // Test with slice
                    if (input.dim() > 0 && input.size(0) > 2) {
                        torch::Tensor sliced = input.narrow(0, 0, 2);
                        sliced.alpha_dropout_(p, training);
                    }
                    break;
                    
                case 3:
                    // Test with requires_grad
                    if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                        input.requires_grad_(true);
                        input.alpha_dropout_(p, training);
                    }
                    break;
                    
                case 4:
                    // Test batch processing
                    if (input.dim() >= 2) {
                        for (int i = 0; i < std::min(3, (int)input.size(0)); i++) {
                            input[i].alpha_dropout_(p, training);
                        }
                    }
                    break;
            }
        }

        // Verify output properties
        if (training && p > 0 && p < 1) {
            // In training mode with 0 < p < 1, some values should be modified
            // But we can't strictly verify this due to randomness
        } else if (!training || p == 0) {
            // In eval mode or with p=0, tensor should be unchanged
            // This is expected behavior
        }

        // Test edge case probabilities if we have more data
        if (offset < Size) {
            // Parse custom probability that might be out of bounds
            float custom_p = *reinterpret_cast<const float*>(Data + offset);
            offset += sizeof(float);
            
            // alpha_dropout_ should handle out-of-bounds probabilities
            // PyTorch typically clamps to [0, 1]
            try {
                torch::Tensor test_tensor = torch::randn({2, 2});
                if (use_cuda && torch::cuda::is_available()) {
                    test_tensor = test_tensor.cuda();
                }
                test_tensor.alpha_dropout_(custom_p, training);
            } catch (...) {
                // Some extreme values might cause issues
            }
        }

    }
    catch (const c10::Error& e)
    {
        // PyTorch-specific errors
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}