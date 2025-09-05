#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for: tensor metadata (2), p value (8), training flag (1)
        if (Size < 11) {
            // Not enough data for even basic parameters
            return 0;
        }

        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dropout probability p (double)
        double p = 0.5; // default
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Normalize p to valid range [0, 1]
            // Use absolute value and modulo to keep in range
            p = std::abs(p);
            if (!std::isfinite(p)) {
                p = 0.5;
            } else {
                // Map to [0, 1] range
                while (p > 1.0) {
                    p = p - std::floor(p);
                }
            }
        }
        
        // Parse training flag
        bool training = true;
        if (offset < Size) {
            training = (Data[offset++] % 2) == 1;
        }
        
        // Try different tensor configurations to maximize coverage
        
        // Test 1: Apply alpha_dropout_ directly
        try {
            torch::Tensor tensor1 = input.clone();
            // alpha_dropout_ modifies tensor in-place
            // The function signature is typically: alpha_dropout_(tensor, p, training)
            tensor1 = torch::alpha_dropout(tensor1, p, training);
            
            // Also test the in-place version if available
            torch::Tensor tensor2 = input.clone();
            tensor2.requires_grad_(false); // Ensure we can modify in-place
            
            // Try with different memory layouts
            if (offset < Size && (Data[offset] % 3) == 0) {
                tensor2 = tensor2.contiguous();
            } else if (offset < Size && (Data[offset] % 3) == 1) {
                // Create non-contiguous tensor by transposing if possible
                if (tensor2.dim() >= 2) {
                    tensor2 = tensor2.transpose(0, 1);
                }
            }
            
            // Apply alpha dropout with different configurations
            torch::Tensor result = torch::alpha_dropout(tensor2, p, training);
            
            // Test edge cases with p values
            if (offset + 1 < Size) {
                // Test with p = 0 (no dropout)
                torch::Tensor tensor3 = input.clone();
                torch::alpha_dropout(tensor3, 0.0, training);
                
                // Test with p = 1 (full dropout) 
                torch::Tensor tensor4 = input.clone();
                torch::alpha_dropout(tensor4, 1.0, training);
            }
            
            // Test with training = false (inference mode)
            torch::Tensor tensor5 = input.clone();
            torch::alpha_dropout(tensor5, p, false);
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid operations
            // Continue execution
        } catch (const std::exception& e) {
            // Other exceptions, log but continue
        }
        
        // Test 2: Try with different tensor properties
        if (offset + 2 < Size) {
            uint8_t test_variant = Data[offset++];
            
            if (test_variant % 4 == 0) {
                // Test with requires_grad
                torch::Tensor grad_tensor = input.clone().requires_grad_(true);
                try {
                    torch::Tensor result = torch::alpha_dropout(grad_tensor, p, training);
                    // Try backward pass if tensor requires grad
                    if (result.requires_grad() && result.numel() > 0) {
                        result.sum().backward();
                    }
                } catch (...) {
                    // Ignore gradient-related errors
                }
            } else if (test_variant % 4 == 1) {
                // Test with different devices if available
                if (torch::cuda::is_available() && input.numel() < 1000000) {
                    try {
                        torch::Tensor cuda_tensor = input.cuda();
                        torch::alpha_dropout(cuda_tensor, p, training);
                    } catch (...) {
                        // CUDA errors are expected if not properly configured
                    }
                }
            } else if (test_variant % 4 == 2) {
                // Test with sparse tensors if applicable
                if (input.dim() == 2 && input.numel() > 0) {
                    try {
                        torch::Tensor sparse = input.to_sparse();
                        torch::alpha_dropout(sparse, p, training);
                    } catch (...) {
                        // Sparse operations might not be supported
                    }
                }
            }
        }
        
        // Test 3: Stress test with extreme shapes
        if (offset + 1 < Size) {
            uint8_t shape_test = Data[offset++];
            
            try {
                if (shape_test % 3 == 0) {
                    // Empty tensor
                    torch::Tensor empty = torch::empty({0}, input.options());
                    torch::alpha_dropout(empty, p, training);
                } else if (shape_test % 3 == 1) {
                    // Scalar tensor
                    torch::Tensor scalar = torch::ones({}, input.options());
                    torch::alpha_dropout(scalar, p, training);
                } else {
                    // High-dimensional tensor with small size
                    std::vector<int64_t> shape(MAX_RANK, 1);
                    torch::Tensor high_dim = torch::ones(shape, input.options());
                    torch::alpha_dropout(high_dim, p, training);
                }
            } catch (...) {
                // Shape-related errors are expected
            }
        }
        
        // Test 4: Multiple sequential operations
        if (input.numel() > 0 && input.numel() < 100000) {
            try {
                torch::Tensor chain = input.clone();
                for (int i = 0; i < 3 && offset + i < Size; ++i) {
                    double p_iter = (Data[offset + i] % 100) / 100.0;
                    chain = torch::alpha_dropout(chain, p_iter, training);
                }
            } catch (...) {
                // Chained operations might accumulate numerical errors
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0; // keep the input
}