#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation
        if (size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        
        // Test signbit operation
        torch::Tensor result = torch::signbit(input);
        
        // Additional testing with different configurations if we have more data
        if (offset < size) {
            // Test with pre-allocated output tensor
            torch::Tensor out_tensor = torch::empty_like(result, torch::dtype(torch::kBool));
            torch::signbit_out(out_tensor, input);
            
            // Verify output consistency
            if (!torch::equal(result, out_tensor)) {
                std::cerr << "Inconsistency between signbit and signbit_out" << std::endl;
            }
        }
        
        // Test edge cases based on remaining fuzzer data
        if (offset + 1 < size) {
            uint8_t edge_case_selector = data[offset++];
            
            switch (edge_case_selector % 8) {
                case 0: {
                    // Test with scalar tensor
                    torch::Tensor scalar = torch::tensor(0.0);
                    torch::signbit(scalar);
                    break;
                }
                case 1: {
                    // Test with negative zero (floating point specific)
                    if (input.is_floating_point()) {
                        torch::Tensor neg_zero = torch::tensor(-0.0).to(input.dtype());
                        torch::Tensor pos_zero = torch::tensor(0.0).to(input.dtype());
                        torch::Tensor neg_result = torch::signbit(neg_zero);
                        torch::Tensor pos_result = torch::signbit(pos_zero);
                        // Negative zero should return true, positive zero false
                    }
                    break;
                }
                case 2: {
                    // Test with infinity values (if floating point)
                    if (input.is_floating_point()) {
                        torch::Tensor inf_tensor = torch::tensor(
                            std::numeric_limits<float>::infinity()
                        ).to(input.dtype());
                        torch::Tensor neg_inf_tensor = torch::tensor(
                            -std::numeric_limits<float>::infinity()
                        ).to(input.dtype());
                        torch::signbit(inf_tensor);
                        torch::signbit(neg_inf_tensor);
                    }
                    break;
                }
                case 3: {
                    // Test with NaN values (if floating point)
                    if (input.is_floating_point()) {
                        torch::Tensor nan_tensor = torch::tensor(
                            std::numeric_limits<float>::quiet_NaN()
                        ).to(input.dtype());
                        torch::signbit(nan_tensor);
                    }
                    break;
                }
                case 4: {
                    // Test with empty tensor
                    torch::Tensor empty = torch::empty({0}, input.options());
                    torch::signbit(empty);
                    break;
                }
                case 5: {
                    // Test with reshaped/viewed tensor (non-contiguous)
                    if (input.numel() > 1) {
                        auto permuted = input.permute({-1});
                        torch::signbit(permuted);
                    }
                    break;
                }
                case 6: {
                    // Test with sliced tensor
                    if (input.numel() > 0) {
                        auto sliced = input.narrow(0, 0, std::min(int64_t(1), input.size(0)));
                        torch::signbit(sliced);
                    }
                    break;
                }
                case 7: {
                    // Test with different memory layouts if multi-dimensional
                    if (input.dim() > 1) {
                        auto transposed = input.transpose(0, -1);
                        torch::signbit(transposed);
                    }
                    break;
                }
            }
        }
        
        // Test with various dtype conversions if we have more data
        if (offset + 1 < size) {
            uint8_t dtype_selector = data[offset++];
            
            // Convert to different dtypes and test signbit
            switch (dtype_selector % 6) {
                case 0:
                    if (input.dtype() != torch::kFloat32) {
                        torch::signbit(input.to(torch::kFloat32));
                    }
                    break;
                case 1:
                    if (input.dtype() != torch::kFloat64) {
                        torch::signbit(input.to(torch::kFloat64));
                    }
                    break;
                case 2:
                    if (input.dtype() != torch::kInt32) {
                        torch::signbit(input.to(torch::kInt32));
                    }
                    break;
                case 3:
                    if (input.dtype() != torch::kInt64) {
                        torch::signbit(input.to(torch::kInt64));
                    }
                    break;
                case 4:
                    if (input.dtype() != torch::kInt8) {
                        torch::signbit(input.to(torch::kInt8));
                    }
                    break;
                case 5:
                    // Test with half precision if supported
                    if (torch::cuda::is_available() || torch::hasCPU()) {
                        if (input.dtype() != torch::kHalf) {
                            torch::signbit(input.to(torch::kHalf));
                        }
                    }
                    break;
            }
        }
        
        // Test batch operations if we have enough elements
        if (input.numel() > 10 && offset + 2 < size) {
            uint8_t batch_size = data[offset++] % 5 + 1;
            uint8_t chunk_size = data[offset++] % 5 + 1;
            
            // Create batched tensor if possible
            auto total_elements = input.numel();
            if (total_elements >= batch_size * chunk_size) {
                std::vector<int64_t> batch_shape = {batch_size, total_elements / batch_size};
                auto batched = input.reshape(batch_shape);
                torch::signbit(batched);
            }
        }
        
        // Test with requires_grad if floating point
        if (input.is_floating_point() && offset < size) {
            uint8_t grad_flag = data[offset++];
            if (grad_flag % 2 == 0) {
                auto grad_input = input.detach().requires_grad_(true);
                auto grad_result = torch::signbit(grad_input);
                // Note: signbit doesn't support backward, but we test the forward pass
            }
        }
        
        // Validate result properties
        if (result.defined()) {
            // Result should always be boolean type
            if (result.dtype() != torch::kBool) {
                std::cerr << "Warning: signbit result is not boolean type" << std::endl;
            }
            
            // Result shape should match input shape
            if (result.sizes() != input.sizes()) {
                std::cerr << "Warning: signbit result shape doesn't match input shape" << std::endl;
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors
        return 0; // Continue fuzzing
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // Discard input for non-PyTorch exceptions
    } catch (...) {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}