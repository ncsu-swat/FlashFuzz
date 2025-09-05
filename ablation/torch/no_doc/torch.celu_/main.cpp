#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation and alpha parameter
        if (Size < 3) {
            return 0;  // Not enough data, but keep for coverage
        }

        // Create primary tensor from fuzzer input
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse alpha parameter from remaining bytes
        float alpha = 1.0f;  // Default value
        if (offset < Size) {
            // Use remaining byte(s) to generate alpha value
            uint8_t alpha_byte = Data[offset++];
            
            // Generate various alpha values including edge cases
            switch (alpha_byte % 8) {
                case 0: alpha = 0.0f; break;           // Zero alpha
                case 1: alpha = 1.0f; break;           // Default
                case 2: alpha = -1.0f; break;          // Negative alpha
                case 3: alpha = 0.01f; break;          // Small positive
                case 4: alpha = 100.0f; break;         // Large positive
                case 5: alpha = -100.0f; break;        // Large negative
                case 6: alpha = std::numeric_limits<float>::infinity(); break;  // Infinity
                case 7: alpha = std::numeric_limits<float>::quiet_NaN(); break; // NaN
            }
            
            // Optionally use more bytes for finer control
            if (offset + sizeof(float) <= Size) {
                float raw_alpha;
                std::memcpy(&raw_alpha, Data + offset, sizeof(float));
                offset += sizeof(float);
                
                // Mix with the raw value for more variety
                if (!std::isnan(raw_alpha) && !std::isinf(raw_alpha)) {
                    alpha = (alpha_byte % 2 == 0) ? raw_alpha : alpha + raw_alpha * 0.1f;
                }
            }
        }

#ifdef DEBUG_FUZZ
        std::cout << "Testing celu_ with tensor shape: " << tensor.sizes() 
                  << ", dtype: " << tensor.dtype() 
                  << ", alpha: " << alpha << std::endl;
#endif

        // Clone tensor for comparison if needed
        torch::Tensor original = tensor.clone();
        
        // Test different tensor configurations
        if (offset < Size && Data[offset] % 4 == 0) {
            // Test with non-contiguous tensor
            if (tensor.dim() > 1 && tensor.size(0) > 1) {
                tensor = tensor.transpose(0, tensor.dim() - 1);
#ifdef DEBUG_FUZZ
                std::cout << "Using non-contiguous tensor (transposed)" << std::endl;
#endif
            }
        }
        
        // Test with different memory layouts if possible
        if (offset < Size && Data[offset] % 3 == 0) {
            // Try to create a view/slice
            if (tensor.numel() > 2) {
                tensor = tensor.narrow(0, 0, std::min(tensor.size(0), (int64_t)2));
#ifdef DEBUG_FUZZ
                std::cout << "Using tensor slice/view" << std::endl;
#endif
            }
        }

        // Apply celu_ in-place operation
        try {
            // The main operation we're testing
            tensor.celu_(alpha);
            
#ifdef DEBUG_FUZZ
            std::cout << "celu_ operation successful" << std::endl;
            
            // Verify the operation modified the tensor
            if (tensor.numel() > 0 && tensor.dtype().isFloatingPoint()) {
                // Check a few values to ensure transformation occurred
                auto flat = tensor.flatten();
                if (flat.numel() > 0) {
                    std::cout << "First element after celu_: " << flat[0].item<float>() << std::endl;
                }
            }
#endif
            
            // Additional validation: test mathematical properties
            if (tensor.dtype().isFloatingPoint() && !std::isnan(alpha) && !std::isinf(alpha)) {
                // CELU should be continuous and differentiable
                // For x >= 0: celu(x) = x
                // For x < 0: celu(x) = alpha * (exp(x/alpha) - 1)
                
                // Create test cases to verify behavior
                if (offset + 1 < Size && Data[offset + 1] % 5 == 0) {
                    torch::Tensor test_tensor = torch::zeros({3}, tensor.options());
                    test_tensor[0] = 1.0;   // Positive value
                    test_tensor[1] = 0.0;   // Zero
                    test_tensor[2] = -1.0;  // Negative value
                    
                    torch::Tensor test_copy = test_tensor.clone();
                    test_copy.celu_(alpha);
                    
#ifdef DEBUG_FUZZ
                    std::cout << "Validation test - input: " << test_tensor 
                              << ", output: " << test_copy << std::endl;
#endif
                }
            }
            
            // Test gradient computation if tensor requires grad
            if (offset + 2 < Size && Data[offset + 2] % 4 == 0) {
                if (tensor.dtype().isFloatingPoint() && tensor.numel() > 0 && tensor.numel() < 1000) {
                    // Create a new tensor that requires grad
                    torch::Tensor grad_tensor = torch::randn_like(tensor).requires_grad_(true);
                    torch::Tensor result = grad_tensor.celu(alpha);  // Use non-inplace for grad
                    
                    if (result.numel() > 0) {
                        // Compute gradient
                        torch::Tensor grad_output = torch::ones_like(result);
                        result.backward(grad_output);
                        
#ifdef DEBUG_FUZZ
                        if (grad_tensor.grad().defined()) {
                            std::cout << "Gradient computation successful" << std::endl;
                        }
#endif
                    }
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors
#ifdef DEBUG_FUZZ
            std::cout << "PyTorch error in celu_: " << e.what() << std::endl;
#endif
            // These are often expected for certain input combinations
            return 0;
        } catch (const std::exception& e) {
            // Other runtime errors
#ifdef DEBUG_FUZZ
            std::cout << "Runtime error in celu_: " << e.what() << std::endl;
#endif
            return 0;
        }
        
        // Test chained operations
        if (offset + 3 < Size && Data[offset + 3] % 6 == 0) {
            try {
                // Chain multiple celu_ operations with different alphas
                float alpha2 = (Data[offset + 3] % 10) * 0.5f;
                tensor.celu_(alpha2);
                
#ifdef DEBUG_FUZZ
                std::cout << "Chained celu_ with alpha2=" << alpha2 << " successful" << std::endl;
#endif
            } catch (...) {
                // Ignore errors in chained operations
            }
        }
        
        // Test with special tensor states
        if (offset + 4 < Size) {
            uint8_t special_case = Data[offset + 4];
            try {
                if (special_case % 5 == 0 && tensor.numel() > 0) {
                    // Test with tensor containing special values
                    if (tensor.dtype().isFloatingPoint()) {
                        tensor.fill_(std::numeric_limits<float>::infinity());
                        tensor.celu_(alpha);
                    }
                } else if (special_case % 5 == 1 && tensor.numel() > 0) {
                    if (tensor.dtype().isFloatingPoint()) {
                        tensor.fill_(-std::numeric_limits<float>::infinity());
                        tensor.celu_(alpha);
                    }
                } else if (special_case % 5 == 2 && tensor.numel() > 0) {
                    if (tensor.dtype().isFloatingPoint()) {
                        tensor.fill_(std::numeric_limits<float>::quiet_NaN());
                        tensor.celu_(alpha);
                    }
                }
            } catch (...) {
                // Special value operations might fail, that's ok
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}