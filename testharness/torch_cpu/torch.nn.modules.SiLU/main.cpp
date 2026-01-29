#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <limits>

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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // SiLU requires floating point tensor
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create SiLU module and apply it
        torch::nn::SiLU silu_module;
        torch::Tensor output = silu_module(input);
        
        // Test functional API
        torch::Tensor output_functional = torch::silu(input);
        
        // Test inplace version on a clone (must not require grad)
        try {
            torch::Tensor input_clone = input.clone().detach();
            torch::silu_(input_clone);
        } catch (...) {
            // Silently ignore inplace failures
        }
        
        // Test with different tensor shapes
        if (offset + 1 < Size) {
            uint8_t shape_type = Data[offset++] % 4;
            torch::Tensor shaped_input;
            
            try {
                if (shape_type == 0) {
                    // Scalar
                    shaped_input = torch::tensor(1.5f);
                } else if (shape_type == 1) {
                    // 1D tensor
                    shaped_input = torch::randn({8});
                } else if (shape_type == 2) {
                    // 2D tensor
                    shaped_input = torch::randn({4, 4});
                } else {
                    // 3D tensor
                    shaped_input = torch::randn({2, 3, 4});
                }
                
                torch::nn::SiLU silu_shaped;
                silu_shaped(shaped_input);
            } catch (...) {
                // Silently ignore shape-related failures
            }
        }
        
        // Test with edge cases
        if (offset + 1 < Size) {
            uint8_t extreme_type = Data[offset++] % 5;
            torch::Tensor extreme_input;
            
            try {
                auto sizes = input.sizes().vec();
                if (sizes.empty()) {
                    sizes = {1};
                }
                
                if (extreme_type == 0) {
                    // Very large positive values
                    extreme_input = torch::full(sizes, 100.0f);
                } else if (extreme_type == 1) {
                    // Very large negative values
                    extreme_input = torch::full(sizes, -100.0f);
                } else if (extreme_type == 2) {
                    // Values near zero
                    extreme_input = torch::full(sizes, 1e-7f);
                } else if (extreme_type == 3) {
                    // NaN values
                    extreme_input = torch::full(sizes, std::numeric_limits<float>::quiet_NaN());
                } else {
                    // Infinity values
                    extreme_input = torch::full(sizes, std::numeric_limits<float>::infinity());
                }
                
                torch::nn::SiLU silu_extreme;
                silu_extreme(extreme_input);
            } catch (...) {
                // Silently ignore extreme value failures
            }
        }
        
        // Test with requires_grad = true
        if (offset + 1 < Size && Data[offset++] % 2 == 0) {
            try {
                torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
                torch::nn::SiLU silu_grad;
                torch::Tensor grad_output = silu_grad(grad_input);
                
                // Trigger backward pass
                if (grad_output.numel() > 0) {
                    torch::Tensor grad_tensor = torch::ones_like(grad_output);
                    grad_output.backward(grad_tensor);
                }
            } catch (...) {
                // Silently ignore gradient-related failures
            }
        }
        
        // Test double precision
        if (offset + 1 < Size && Data[offset++] % 3 == 0) {
            try {
                torch::Tensor double_input = input.to(torch::kFloat64);
                torch::nn::SiLU silu_double;
                silu_double(double_input);
            } catch (...) {
                // Silently ignore dtype conversion failures
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