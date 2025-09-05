#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }

        // Create primary tensor from fuzzer input
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse diagonal offset parameter if we have more data
        int64_t diagonal = 0;
        if (offset < Size) {
            // Use remaining bytes to determine diagonal offset
            uint8_t diag_byte = Data[offset++];
            // Map to reasonable diagonal range [-10, 10]
            diagonal = static_cast<int64_t>(diag_byte % 21) - 10;
        }

        // Test torch.diag with various tensor configurations
        try {
            // Basic diag operation
            torch::Tensor result = torch::diag(tensor, diagonal);
            
            // Verify result is valid
            if (result.numel() > 0) {
                // Force computation to catch lazy evaluation issues
                result.sum().item<float>();
            }
            
            // Additional operations to increase coverage
            if (tensor.dim() == 1) {
                // For 1D input, result should be 2D diagonal matrix
                if (result.dim() == 2) {
                    // Test extracting diagonal back
                    torch::Tensor extracted = torch::diag(result, diagonal);
                    extracted.sum().item<float>();
                }
            } else if (tensor.dim() == 2) {
                // For 2D input, result should be 1D
                if (result.dim() == 1) {
                    // Test creating diagonal matrix from extracted diagonal
                    torch::Tensor recreated = torch::diag(result, diagonal);
                    recreated.sum().item<float>();
                }
            }
            
            // Test edge cases with different diagonal values
            if (offset < Size && tensor.dim() <= 2) {
                // Try multiple diagonal offsets
                for (int i = 0; i < 3 && offset < Size; ++i) {
                    uint8_t extra_diag = Data[offset++];
                    int64_t test_diagonal = static_cast<int64_t>(extra_diag % 41) - 20;
                    
                    try {
                        torch::Tensor diag_result = torch::diag(tensor, test_diagonal);
                        if (diag_result.numel() > 0) {
                            diag_result.sum().item<float>();
                        }
                    } catch (const c10::Error& e) {
                        // Some diagonal values may be out of bounds for the tensor shape
                        // This is expected behavior, continue testing
                    }
                }
            }
            
            // Test with different tensor properties
            if (tensor.numel() > 0 && offset < Size) {
                uint8_t op_selector = Data[offset++];
                
                // Test with contiguous/non-contiguous tensors
                if (op_selector & 0x01) {
                    if (!tensor.is_contiguous()) {
                        torch::Tensor cont_result = torch::diag(tensor.contiguous(), diagonal);
                        cont_result.sum().item<float>();
                    }
                }
                
                // Test with transposed tensor (if 2D)
                if ((op_selector & 0x02) && tensor.dim() == 2) {
                    torch::Tensor transposed = tensor.transpose(0, 1);
                    torch::Tensor trans_result = torch::diag(transposed, diagonal);
                    trans_result.sum().item<float>();
                }
                
                // Test with different memory layouts
                if ((op_selector & 0x04) && tensor.dim() == 2) {
                    // Create a view with different strides
                    if (tensor.size(0) > 1 && tensor.size(1) > 1) {
                        torch::Tensor sliced = tensor.slice(0, 0, tensor.size(0), 2);
                        if (sliced.numel() > 0) {
                            try {
                                torch::Tensor slice_result = torch::diag(sliced, diagonal);
                                slice_result.sum().item<float>();
                            } catch (const c10::Error& e) {
                                // Slicing might create incompatible shapes
                            }
                        }
                    }
                }
                
                // Test with reshaped tensors
                if ((op_selector & 0x08) && tensor.numel() > 1) {
                    // Try to reshape to different valid shapes
                    int64_t numel = tensor.numel();
                    
                    // Try square matrix if possible
                    int64_t sqrt_n = static_cast<int64_t>(std::sqrt(numel));
                    if (sqrt_n * sqrt_n == numel) {
                        torch::Tensor square = tensor.reshape({sqrt_n, sqrt_n});
                        torch::Tensor square_diag = torch::diag(square, diagonal);
                        square_diag.sum().item<float>();
                    }
                    
                    // Try vector
                    torch::Tensor vec = tensor.reshape({numel});
                    torch::Tensor vec_diag = torch::diag(vec, diagonal);
                    vec_diag.sum().item<float>();
                }
            }
            
            // Test with zero-dimensional tensors (scalars)
            if (tensor.dim() == 0) {
                try {
                    // This might throw, but we want to test the behavior
                    torch::Tensor scalar_diag = torch::diag(tensor, diagonal);
                    scalar_diag.sum().item<float>();
                } catch (const c10::Error& e) {
                    // Expected for scalar inputs
                }
            }
            
            // Test with empty tensors
            if (tensor.numel() == 0) {
                torch::Tensor empty_diag = torch::diag(tensor, diagonal);
                // Empty tensor operations should still be valid
                empty_diag.sum().item<float>();
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors (like dimension mismatches) are expected
            // Continue fuzzing
        } catch (const std::runtime_error& e) {
            // Runtime errors from tensor operations
            // Continue fuzzing
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}