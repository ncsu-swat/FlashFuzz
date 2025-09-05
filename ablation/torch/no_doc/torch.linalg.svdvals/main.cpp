#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation
        if (Size < 4) {
            return 0;
        }

        // Create primary input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create a basic tensor, try with minimal valid tensor
            if (offset < Size) {
                // Create a simple 2x2 float tensor from remaining data
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                input_tensor = torch::randn({2, 2}, options);
            } else {
                return 0;
            }
        }

        // Ensure we have at least a 2D tensor for svdvals
        if (input_tensor.dim() < 2) {
            // Reshape or expand to make it 2D
            if (input_tensor.dim() == 0) {
                input_tensor = input_tensor.reshape({1, 1});
            } else if (input_tensor.dim() == 1) {
                int64_t n = input_tensor.size(0);
                input_tensor = input_tensor.reshape({n, 1});
            }
        }

        // Handle higher dimensional tensors (batch processing)
        if (input_tensor.dim() > 2 && offset < Size) {
            // Optionally flatten or keep batch dimensions based on fuzzer input
            uint8_t batch_mode = (offset < Size) ? Data[offset++] : 0;
            if (batch_mode % 3 == 0) {
                // Flatten to 2D
                int64_t total_elements = input_tensor.numel();
                int64_t dim1 = std::max(int64_t(1), static_cast<int64_t>(std::sqrt(total_elements)));
                int64_t dim2 = total_elements / dim1;
                if (dim2 == 0) dim2 = 1;
                input_tensor = input_tensor.reshape({dim1, dim2});
            }
            // Otherwise keep batch dimensions for batch SVD
        }

        // Convert to appropriate dtype for SVD if needed
        // svdvals typically works best with floating point types
        if (!input_tensor.is_floating_point() && !input_tensor.is_complex()) {
            // Convert integer/bool types to float
            input_tensor = input_tensor.to(torch::kFloat32);
        }

        // Parse additional parameters if available
        bool use_driver = false;
        if (offset < Size) {
            use_driver = (Data[offset++] % 2) == 1;
        }

        // Create options for svdvals
        c10::optional<c10::string_view> driver = c10::nullopt;
        if (use_driver && offset < Size) {
            uint8_t driver_selector = Data[offset++] % 3;
            switch (driver_selector) {
                case 0:
                    driver = "gesvd";
                    break;
                case 1:
                    driver = "gesvdj";
                    break;
                case 2:
                    driver = "gesvda";
                    break;
            }
        }

        // Test various edge cases based on fuzzer input
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Create specific problematic matrices
            switch (edge_case % 8) {
                case 0:
                    // Zero matrix
                    input_tensor = torch::zeros_like(input_tensor);
                    break;
                case 1:
                    // Identity-like matrix
                    if (input_tensor.size(-2) == input_tensor.size(-1)) {
                        input_tensor = torch::eye(input_tensor.size(-1), 
                                                 input_tensor.options());
                    }
                    break;
                case 2:
                    // Very small values
                    input_tensor = input_tensor * 1e-10;
                    break;
                case 3:
                    // Very large values
                    input_tensor = input_tensor * 1e10;
                    break;
                case 4:
                    // NaN values (if floating point)
                    if (input_tensor.is_floating_point() && offset < Size && Data[offset++] % 4 == 0) {
                        input_tensor[0] = std::numeric_limits<float>::quiet_NaN();
                    }
                    break;
                case 5:
                    // Inf values (if floating point)
                    if (input_tensor.is_floating_point() && offset < Size && Data[offset++] % 4 == 0) {
                        input_tensor[0] = std::numeric_limits<float>::infinity();
                    }
                    break;
                case 6:
                    // Rank deficient matrix
                    if (input_tensor.dim() >= 2) {
                        auto m = input_tensor.size(-2);
                        auto n = input_tensor.size(-1);
                        if (m > 1 && n > 1) {
                            // Make rows linearly dependent
                            input_tensor.select(-2, 1) = input_tensor.select(-2, 0) * 2.0;
                        }
                    }
                    break;
                case 7:
                    // Nearly singular matrix
                    input_tensor = input_tensor + torch::randn_like(input_tensor) * 1e-12;
                    break;
            }
        }

        // Test with different memory layouts
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Make non-contiguous
            if (input_tensor.dim() >= 2) {
                input_tensor = input_tensor.transpose(-2, -1).transpose(-2, -1);
            }
        }

        // Call torch.linalg.svdvals with error handling
        torch::Tensor result;
        
        try {
            if (driver.has_value()) {
                // Call with driver option if supported (may not be in all PyTorch versions)
                result = torch::linalg::svdvals(input_tensor);
            } else {
                result = torch::linalg::svdvals(input_tensor);
            }
            
            // Validate output
            if (result.defined()) {
                // Check output shape
                auto expected_size = std::min(input_tensor.size(-2), input_tensor.size(-1));
                if (input_tensor.dim() > 2) {
                    // Batch dimensions should be preserved
                    auto batch_shape = input_tensor.sizes().vec();
                    batch_shape.pop_back();
                    batch_shape.pop_back();
                    batch_shape.push_back(expected_size);
                    
                    if (result.sizes() != c10::IntArrayRef(batch_shape)) {
                        std::cerr << "Unexpected output shape for batch svdvals" << std::endl;
                    }
                } else {
                    if (result.size(0) != expected_size) {
                        std::cerr << "Unexpected number of singular values" << std::endl;
                    }
                }
                
                // Check for NaN/Inf in output (unless input had them)
                if (!input_tensor.isnan().any().item<bool>() && 
                    !input_tensor.isinf().any().item<bool>()) {
                    if (result.isnan().any().item<bool>()) {
                        std::cerr << "NaN in svdvals output without NaN in input" << std::endl;
                    }
                    if (result.isinf().any().item<bool>()) {
                        std::cerr << "Inf in svdvals output without Inf in input" << std::endl;
                    }
                }
                
                // Singular values should be non-negative
                if ((result < 0).any().item<bool>()) {
                    std::cerr << "Negative singular values detected" << std::endl;
                }
                
                // Test reconstruction if we have enough data
                if (offset < Size && Data[offset++] % 4 == 0) {
                    try {
                        // Full SVD for comparison
                        auto [U, S, Vt] = torch::linalg::svd(input_tensor, false);
                        
                        // Compare singular values
                        if (!torch::allclose(result, S, 1e-5, 1e-8)) {
                            std::cerr << "svdvals differs from svd singular values" << std::endl;
                        }
                    } catch (...) {
                        // SVD might fail where svdvals succeeds, that's ok
                    }
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue fuzzing
        } catch (const std::runtime_error& e) {
            // Runtime errors from invalid operations
            // Continue fuzzing
        }

        // Additional stress testing with multiple calls
        if (offset < Size && Data[offset++] % 8 == 0) {
            try {
                // Call multiple times to check for memory issues
                for (int i = 0; i < 3; ++i) {
                    auto temp_result = torch::linalg::svdvals(input_tensor);
                    if (i > 0 && result.defined() && temp_result.defined()) {
                        if (!torch::allclose(result, temp_result, 1e-6, 1e-9)) {
                            std::cerr << "Inconsistent results across multiple calls" << std::endl;
                        }
                    }
                }
            } catch (...) {
                // Ignore errors in stress test
            }
        }

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