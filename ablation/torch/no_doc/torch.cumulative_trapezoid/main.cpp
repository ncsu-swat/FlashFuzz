#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for control flow
        if (Size < 4) {
            return 0;
        }

        // Parse control bytes
        uint8_t use_x_tensor = Data[offset++];
        uint8_t dim_selector = Data[offset++];
        
        // Create primary tensor y
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Early exit if y has no dimensions (scalar)
        if (y.dim() == 0) {
            // cumulative_trapezoid requires at least 1 dimension
            y = y.unsqueeze(0);
        }
        
        // Select dimension for integration (modulo to stay in bounds)
        int64_t dim = dim_selector % y.dim();
        
        // Optionally create x tensor (sample points)
        torch::Tensor result;
        
        if (use_x_tensor % 3 == 0 && offset < Size) {
            // Create x tensor for sample points
            try {
                torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
                
                // x must be 1-D or match y's shape along dim
                if (x.dim() > 0) {
                    // Try different x configurations based on fuzzer input
                    uint8_t x_config = (offset < Size) ? Data[offset++] : 0;
                    
                    if (x_config % 3 == 0) {
                        // Make x 1-D with size matching y.size(dim)
                        if (y.size(dim) > 0) {
                            x = x.flatten().slice(0, 0, y.size(dim));
                            if (x.numel() < y.size(dim)) {
                                // Pad with ones if needed
                                auto pad_size = y.size(dim) - x.numel();
                                x = torch::cat({x, torch::ones({pad_size}, x.options())});
                            }
                        }
                    } else if (x_config % 3 == 1) {
                        // Try to broadcast x to match y along dim
                        auto y_shape = y.sizes().vec();
                        for (int i = 0; i < y.dim(); ++i) {
                            if (i != dim) {
                                y_shape[i] = 1;
                            }
                        }
                        x = x.reshape({-1}).slice(0, 0, y.size(dim));
                        if (x.numel() < y.size(dim)) {
                            x = torch::cat({x, torch::ones({y.size(dim) - x.numel()}, x.options())});
                        }
                        x = x.reshape(y_shape);
                    } else {
                        // Use x as-is and let the operation handle shape matching
                        // This may throw, which is fine for fuzzing
                    }
                }
                
                // Call cumulative_trapezoid with x
                result = torch::cumulative_trapezoid(y, x, dim);
                
            } catch (const c10::Error& e) {
                // x tensor creation or usage failed, fall back to default
                result = torch::cumulative_trapezoid(y, dim);
            } catch (const std::exception& e) {
                // Other x-related errors, fall back
                result = torch::cumulative_trapezoid(y, dim);
            }
        } else if (use_x_tensor % 3 == 1) {
            // Use scalar dx (spacing between samples)
            double dx = 1.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&dx, Data + offset, sizeof(double));
                offset += sizeof(double);
                // Normalize dx to reasonable range
                dx = std::abs(dx);
                if (!std::isfinite(dx) || dx == 0.0) {
                    dx = 1.0;
                } else if (dx > 1e6) {
                    dx = std::fmod(dx, 1000.0) + 1.0;
                }
            }
            result = torch::cumulative_trapezoid(y, dx, dim);
        } else {
            // Default: unit spacing
            result = torch::cumulative_trapezoid(y, dim);
        }
        
        // Additional operations to increase coverage
        if (offset < Size) {
            uint8_t post_op = Data[offset++];
            
            if (post_op % 5 == 0 && result.numel() > 0) {
                // Test with different memory layouts
                if (!result.is_contiguous()) {
                    result = result.contiguous();
                }
            } else if (post_op % 5 == 1) {
                // Test with transposed result
                if (result.dim() >= 2) {
                    result = result.transpose(0, 1);
                }
            } else if (post_op % 5 == 2) {
                // Test with view operations
                if (result.numel() > 0) {
                    result = result.view({-1});
                }
            } else if (post_op % 5 == 3) {
                // Test backward compatibility - compute again with different dim
                if (y.dim() > 1) {
                    int64_t alt_dim = (dim + 1) % y.dim();
                    auto result2 = torch::cumulative_trapezoid(y, alt_dim);
                    // Just access to ensure computation
                    (void)result2.sum();
                }
            } else if (post_op % 5 == 4) {
                // Test with complex tensors if applicable
                if (y.is_floating_point() && !y.is_complex()) {
                    auto y_complex = torch::complex(y, torch::zeros_like(y));
                    auto result_complex = torch::cumulative_trapezoid(y_complex, dim);
                    (void)result_complex.real();
                }
            }
        }
        
        // Force computation by accessing result
        if (result.numel() > 0) {
            auto sum = result.sum();
            (void)sum;
        }
        
        // Test edge cases based on remaining fuzzer input
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            if (edge_case % 4 == 0) {
                // Test with empty tensor
                auto empty = torch::empty({0}, y.options());
                try {
                    auto empty_result = torch::cumulative_trapezoid(empty, 0);
                    (void)empty_result;
                } catch (...) {
                    // Expected to potentially fail
                }
            } else if (edge_case % 4 == 1) {
                // Test with single element tensor
                auto single = torch::ones({1}, y.options());
                auto single_result = torch::cumulative_trapezoid(single, 0);
                (void)single_result.item();
            } else if (edge_case % 4 == 2 && y.dim() > 0) {
                // Test with negative dimension
                int64_t neg_dim = -(dim + 1);
                auto neg_result = torch::cumulative_trapezoid(y, neg_dim);
                (void)neg_result.sum();
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        // Catch any other unexpected exceptions
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}