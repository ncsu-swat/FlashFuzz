#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes: dtype, rank, and operation mode
        if (Size < 3) {
            return 0;
        }

        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get operation mode byte if available
        uint8_t op_mode = 0;
        if (offset < Size) {
            op_mode = Data[offset++];
        }
        
        // Test torch::special::round with various configurations
        try {
            // Basic round operation
            torch::Tensor result = torch::special::round(input_tensor);
            
            // Verify result has same shape as input
            if (result.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch after round operation" << std::endl;
            }
            
            // Test with different tensor properties based on op_mode
            switch (op_mode % 8) {
                case 0: {
                    // Test with contiguous tensor
                    if (!input_tensor.is_contiguous()) {
                        auto contig = input_tensor.contiguous();
                        result = torch::special::round(contig);
                    }
                    break;
                }
                case 1: {
                    // Test with non-contiguous tensor (transpose)
                    if (input_tensor.dim() >= 2) {
                        auto transposed = input_tensor.transpose(0, 1);
                        result = torch::special::round(transposed);
                    }
                    break;
                }
                case 2: {
                    // Test with view
                    if (input_tensor.numel() > 0) {
                        auto viewed = input_tensor.view({-1});
                        result = torch::special::round(viewed);
                    }
                    break;
                }
                case 3: {
                    // Test with slice
                    if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
                        auto sliced = input_tensor.slice(0, 0, 1);
                        result = torch::special::round(sliced);
                    }
                    break;
                }
                case 4: {
                    // Test with requires_grad for floating point tensors
                    if (input_tensor.is_floating_point() && !input_tensor.requires_grad()) {
                        input_tensor.requires_grad_(true);
                        result = torch::special::round(input_tensor);
                        // Test backward if possible
                        if (result.requires_grad() && result.numel() > 0) {
                            try {
                                auto grad_output = torch::ones_like(result);
                                result.backward(grad_output);
                            } catch (...) {
                                // Gradient computation might fail for round, which is expected
                            }
                        }
                    }
                    break;
                }
                case 5: {
                    // Test with special values (if floating point)
                    if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
                        // Create tensor with special values
                        auto special_vals = torch::empty_like(input_tensor);
                        special_vals.fill_(std::numeric_limits<float>::infinity());
                        result = torch::special::round(special_vals);
                        
                        special_vals.fill_(-std::numeric_limits<float>::infinity());
                        result = torch::special::round(special_vals);
                        
                        special_vals.fill_(std::numeric_limits<float>::quiet_NaN());
                        result = torch::special::round(special_vals);
                    }
                    break;
                }
                case 6: {
                    // Test with edge values (0.5, -0.5, etc.)
                    if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
                        auto edge_vals = torch::empty_like(input_tensor);
                        edge_vals.fill_(0.5);
                        result = torch::special::round(edge_vals);
                        
                        edge_vals.fill_(-0.5);
                        result = torch::special::round(edge_vals);
                        
                        edge_vals.fill_(1.5);
                        result = torch::special::round(edge_vals);
                        
                        edge_vals.fill_(-1.5);
                        result = torch::special::round(edge_vals);
                    }
                    break;
                }
                case 7: {
                    // Test with out parameter
                    torch::Tensor out_tensor = torch::empty_like(input_tensor);
                    torch::special::round_out(out_tensor, input_tensor);
                    
                    // Test with mismatched out tensor (should resize)
                    if (input_tensor.numel() > 0) {
                        torch::Tensor small_out = torch::empty({1});
                        try {
                            torch::special::round_out(small_out, input_tensor);
                        } catch (...) {
                            // Expected to fail or resize
                        }
                    }
                    break;
                }
            }
            
            // Additional edge case testing based on remaining bytes
            if (offset < Size) {
                uint8_t extra_test = Data[offset++];
                
                if (extra_test % 4 == 0 && input_tensor.dim() > 0) {
                    // Test with permuted tensor
                    std::vector<int64_t> dims;
                    for (int64_t i = input_tensor.dim() - 1; i >= 0; --i) {
                        dims.push_back(i);
                    }
                    auto permuted = input_tensor.permute(dims);
                    result = torch::special::round(permuted);
                }
                else if (extra_test % 4 == 1) {
                    // Test with expanded tensor
                    if (input_tensor.dim() > 0) {
                        auto sizes = input_tensor.sizes().vec();
                        for (auto& s : sizes) {
                            if (s == 1 && extra_test > 127) {
                                s = 2;
                                break;
                            }
                        }
                        try {
                            auto expanded = input_tensor.expand(sizes);
                            result = torch::special::round(expanded);
                        } catch (...) {
                            // Expansion might fail
                        }
                    }
                }
                else if (extra_test % 4 == 2) {
                    // Test with narrow tensor
                    if (input_tensor.dim() > 0 && input_tensor.size(0) > 2) {
                        auto narrowed = input_tensor.narrow(0, 1, 1);
                        result = torch::special::round(narrowed);
                    }
                }
                else {
                    // Test with unfold if possible
                    if (input_tensor.dim() > 0 && input_tensor.size(0) >= 3) {
                        try {
                            auto unfolded = input_tensor.unfold(0, 2, 1);
                            result = torch::special::round(unfolded);
                        } catch (...) {
                            // Unfold might fail with certain parameters
                        }
                    }
                }
            }
            
            // Test chained operations
            if (offset < Size && Data[offset++] % 2 == 0) {
                try {
                    auto result2 = torch::special::round(torch::special::round(input_tensor));
                    // Double rounding should give same result as single rounding
                    if (torch::allclose(result, result2, 1e-5, 1e-8)) {
                        // Expected behavior
                    }
                } catch (...) {
                    // Chain might fail
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for some edge cases
            return 0;
        } catch (const std::exception& e) {
            // Log unexpected standard exceptions
            std::cout << "Exception in round operation: " << e.what() << std::endl;
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