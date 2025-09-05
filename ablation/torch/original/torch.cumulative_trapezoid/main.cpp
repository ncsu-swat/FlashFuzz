#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <optional>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 4) {
        // Need at least 4 bytes for basic configuration
        return 0;
    }

    try
    {
        size_t offset = 0;

        // Parse configuration flags
        uint8_t config_byte = data[offset++];
        bool use_x_tensor = (config_byte & 0x01) != 0;
        bool use_dx = (config_byte & 0x02) != 0;
        bool use_custom_dim = (config_byte & 0x04) != 0;
        
        // If both x and dx are specified, PyTorch should handle the conflict
        // We don't prevent it to test error handling

        // Create the main tensor y
        torch::Tensor y = fuzzer_utils::createTensor(data, size, offset);
        
        if (y.numel() == 0 && y.dim() > 0) {
            // Skip empty tensors with dimensions as they may cause issues
            return 0;
        }

        // Prepare optional x tensor
        std::optional<torch::Tensor> x;
        if (use_x_tensor && offset < size) {
            try {
                torch::Tensor x_tensor = fuzzer_utils::createTensor(data, size, offset);
                
                // x should have compatible shape with y along the integration dimension
                // Let PyTorch validate this - we want to test error handling too
                x = x_tensor;
            } catch (const std::exception& e) {
                // If we can't create x tensor, proceed without it
                use_x_tensor = false;
            }
        }

        // Parse dx value if needed
        std::optional<double> dx;
        if (use_dx && offset + sizeof(double) <= size) {
            double dx_val;
            std::memcpy(&dx_val, data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Sanitize dx to avoid extreme values
            if (std::isfinite(dx_val)) {
                // Clamp to reasonable range
                dx_val = std::max(-1e6, std::min(1e6, dx_val));
                dx = dx_val;
            }
        }

        // Parse dimension
        int64_t dim = -1; // Default to last dimension
        if (use_custom_dim && offset < size) {
            uint8_t dim_byte = data[offset++];
            if (y.dim() > 0) {
                // Map to valid dimension range [-rank, rank-1]
                int64_t rank = y.dim();
                dim = static_cast<int64_t>(dim_byte) % (2 * rank) - rank;
            }
        }

        // Call cumulative_trapezoid with different parameter combinations
        torch::Tensor result;
        
        try {
            if (use_x_tensor && x.has_value()) {
                // Case 1: Using x tensor for spacing
                result = torch::cumulative_trapezoid(y, x.value(), dim);
            } else if (use_dx && dx.has_value()) {
                // Case 2: Using constant dx spacing
                result = torch::cumulative_trapezoid(y, c10::nullopt, dx.value(), dim);
            } else {
                // Case 3: Default spacing (implicitly 1)
                result = torch::cumulative_trapezoid(y, c10::nullopt, c10::nullopt, dim);
            }

            // Validate result properties
            if (result.defined()) {
                // Check that result has one less element along the integration dimension
                if (y.dim() > 0) {
                    int64_t actual_dim = (dim < 0) ? (y.dim() + dim) : dim;
                    if (actual_dim >= 0 && actual_dim < y.dim()) {
                        auto expected_size = y.sizes().vec();
                        expected_size[actual_dim] = std::max(int64_t(0), y.size(actual_dim) - 1);
                        
                        if (result.sizes() != c10::IntArrayRef(expected_size)) {
                            std::cerr << "Unexpected output shape: " << result.sizes() 
                                     << " vs expected " << c10::IntArrayRef(expected_size) << std::endl;
                        }
                    }
                }

                // Test various tensor operations on the result
                if (result.numel() > 0) {
                    // Check for NaN/Inf in result
                    if (result.dtype().isFloatingPoint() || result.dtype().isComplex()) {
                        auto has_nan = torch::any(torch::isnan(result));
                        auto has_inf = torch::any(torch::isinf(result));
                        
                        // These aren't necessarily errors, but good to know during fuzzing
                        if (has_nan.item<bool>()) {
                            // NaN in result - could be from NaN in input or overflow
                        }
                        if (has_inf.item<bool>()) {
                            // Inf in result - could be from Inf in input or overflow
                        }
                    }

                    // Try some basic operations to ensure result tensor is valid
                    auto sum = result.sum();
                    auto mean = result.mean();
                    
                    // Test gradient computation if applicable
                    if (result.requires_grad() && result.dtype().isFloatingPoint()) {
                        try {
                            auto grad_result = result.sum();
                            grad_result.backward();
                        } catch (const std::exception& e) {
                            // Gradient computation failed - not necessarily an error
                        }
                    }
                }
            }

            // Additional edge case testing with the same input
            if (y.dim() > 1 && offset < size) {
                // Test with different dimensions on the same tensor
                for (int64_t test_dim = 0; test_dim < y.dim(); ++test_dim) {
                    try {
                        auto result2 = torch::cumulative_trapezoid(y, c10::nullopt, c10::nullopt, test_dim);
                    } catch (const std::exception& e) {
                        // Some dimensions might fail for certain tensor configurations
                    }
                }
            }

            // Test with modified tensors
            if (y.numel() > 0 && y.dtype().isFloatingPoint()) {
                // Test with requires_grad
                auto y_grad = y.detach().requires_grad_(true);
                try {
                    auto result_grad = torch::cumulative_trapezoid(y_grad, c10::nullopt, c10::nullopt, dim);
                    if (result_grad.requires_grad() && result_grad.numel() > 0) {
                        result_grad.sum().backward();
                    }
                } catch (const std::exception& e) {
                    // Autograd might fail for some configurations
                }
            }

        } catch (const c10::Error& e) {
            // PyTorch C++ exceptions are expected for invalid inputs
            // This is normal behavior during fuzzing
            return 0;
        } catch (const std::exception& e) {
            // Other exceptions might indicate actual issues
            std::cout << "Unexpected exception in cumulative_trapezoid: " << e.what() << std::endl;
            return 0;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}