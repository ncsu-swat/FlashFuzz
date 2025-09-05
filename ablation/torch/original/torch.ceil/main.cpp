#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    // Minimum size check - need at least a few bytes for basic tensor creation
    if (size < 4) {
        return 0;  // Not enough data to create meaningful test
    }

    try
    {
        size_t offset = 0;
        
        // Create primary input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
        
        // Test 1: Basic ceil operation
        torch::Tensor result = torch::ceil(input);
        
        // Verify output shape matches input
        if (result.sizes() != input.sizes()) {
            std::cerr << "Unexpected shape change in ceil operation" << std::endl;
            return -1;
        }
        
        // Test 2: In-place ceil operation if tensor is floating point
        if (input.is_floating_point()) {
            torch::Tensor input_copy = input.clone();
            input_copy.ceil_();
            
            // Verify in-place and out-of-place produce same results
            if (!torch::allclose(result, input_copy, 1e-5, 1e-8)) {
                std::cerr << "In-place and out-of-place ceil differ" << std::endl;
            }
        }
        
        // Test 3: ceil with out parameter
        if (offset < size) {
            // Try to create an output tensor if we have more data
            try {
                torch::Tensor out_tensor = torch::empty_like(input);
                torch::ceil_out(out_tensor, input);
                
                // Verify out parameter version produces same result
                if (!torch::allclose(result, out_tensor, 1e-5, 1e-8)) {
                    std::cerr << "ceil with out parameter differs from regular ceil" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Some dtype/device combinations might not support out parameter
                // Continue testing
            }
        }
        
        // Test 4: Edge cases based on remaining fuzzer data
        if (offset < size) {
            uint8_t edge_case_selector = data[offset++];
            
            switch (edge_case_selector % 8) {
                case 0: {
                    // Test with zero tensor
                    torch::Tensor zero_tensor = torch::zeros_like(input);
                    torch::Tensor zero_result = torch::ceil(zero_tensor);
                    if (!torch::all(zero_result == zero_tensor).item<bool>()) {
                        std::cerr << "ceil(0) != 0 detected" << std::endl;
                    }
                    break;
                }
                case 1: {
                    // Test with ones tensor
                    torch::Tensor ones_tensor = torch::ones_like(input);
                    torch::Tensor ones_result = torch::ceil(ones_tensor);
                    if (!torch::all(ones_result == ones_tensor).item<bool>()) {
                        std::cerr << "ceil(1) != 1 detected" << std::endl;
                    }
                    break;
                }
                case 2: {
                    // Test with negative values
                    if (input.is_floating_point()) {
                        torch::Tensor neg_tensor = -torch::abs(input);
                        torch::Tensor neg_result = torch::ceil(neg_tensor);
                        // Just execute, don't validate specific values
                    }
                    break;
                }
                case 3: {
                    // Test with infinity values (if floating point)
                    if (input.is_floating_point()) {
                        torch::Tensor inf_tensor = torch::full_like(input, std::numeric_limits<float>::infinity());
                        torch::Tensor inf_result = torch::ceil(inf_tensor);
                        // ceil(inf) should be inf
                        if (!torch::all(torch::isinf(inf_result)).item<bool>()) {
                            std::cerr << "ceil(inf) != inf detected" << std::endl;
                        }
                    }
                    break;
                }
                case 4: {
                    // Test with NaN values (if floating point)
                    if (input.is_floating_point()) {
                        torch::Tensor nan_tensor = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                        torch::Tensor nan_result = torch::ceil(nan_tensor);
                        // ceil(nan) should be nan
                        if (!torch::all(torch::isnan(nan_result)).item<bool>()) {
                            std::cerr << "ceil(nan) != nan detected" << std::endl;
                        }
                    }
                    break;
                }
                case 5: {
                    // Test with very small values
                    if (input.is_floating_point()) {
                        torch::Tensor small_tensor = input * 1e-10;
                        torch::Tensor small_result = torch::ceil(small_tensor);
                        // Just execute to test numerical stability
                    }
                    break;
                }
                case 6: {
                    // Test with mixed positive/negative values
                    if (input.numel() > 1 && input.is_floating_point()) {
                        torch::Tensor mixed = input.clone();
                        // Make half negative
                        auto flat = mixed.flatten();
                        int64_t half = flat.size(0) / 2;
                        if (half > 0) {
                            flat.slice(0, 0, half).mul_(-1);
                        }
                        torch::Tensor mixed_result = torch::ceil(mixed);
                        // Just execute
                    }
                    break;
                }
                case 7: {
                    // Test with already ceiled values
                    torch::Tensor already_ceiled = torch::ceil(input);
                    torch::Tensor double_ceiled = torch::ceil(already_ceiled);
                    // ceil(ceil(x)) should equal ceil(x)
                    if (!torch::allclose(already_ceiled, double_ceiled, 1e-5, 1e-8)) {
                        std::cerr << "ceil(ceil(x)) != ceil(x) detected" << std::endl;
                    }
                    break;
                }
            }
        }
        
        // Test 5: Different tensor layouts if we have more data
        if (offset < size && size - offset >= 2) {
            uint8_t layout_selector = data[offset++];
            
            if (layout_selector % 3 == 0 && input.dim() >= 2) {
                // Test with transposed tensor
                torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                torch::Tensor trans_result = torch::ceil(transposed);
                // Verify shape is preserved
                if (trans_result.sizes() != transposed.sizes()) {
                    std::cerr << "Shape not preserved for transposed tensor" << std::endl;
                }
            } else if (layout_selector % 3 == 1 && input.numel() > 0) {
                // Test with reshaped tensor
                torch::Tensor reshaped = input.reshape({-1});
                torch::Tensor reshape_result = torch::ceil(reshaped);
                // Verify element count is preserved
                if (reshape_result.numel() != reshaped.numel()) {
                    std::cerr << "Element count not preserved for reshaped tensor" << std::endl;
                }
            } else if (layout_selector % 3 == 2 && input.dim() > 0) {
                // Test with squeezed tensor
                torch::Tensor squeezed = input.squeeze();
                torch::Tensor squeeze_result = torch::ceil(squeezed);
                // Just execute
            }
        }
        
        // Test 6: Test with different memory formats if applicable
        if (input.dim() == 4 && input.is_floating_point()) {
            // Test with channels_last memory format (NCHW -> NHWC)
            try {
                torch::Tensor channels_last = input.to(torch::MemoryFormat::ChannelsLast);
                torch::Tensor cl_result = torch::ceil(channels_last);
                // Result should preserve memory format
                if (!cl_result.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
                    std::cerr << "Memory format not preserved for channels_last tensor" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Some configurations might not support channels_last
            }
        }
        
        // Test 7: Verify mathematical properties
        if (input.is_floating_point() && input.numel() > 0) {
            // Property: ceil(x) >= x for all x
            torch::Tensor diff = result - input;
            if (!torch::all(diff >= 0).item<bool>()) {
                std::cerr << "Mathematical property violated: ceil(x) < x found" << std::endl;
                return -1;
            }
            
            // Property: ceil(x) - x < 1 for all x
            if (!torch::all(diff < 1).item<bool>()) {
                std::cerr << "Mathematical property violated: ceil(x) - x >= 1 found" << std::endl;
                return -1;
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors - these are expected for some inputs
        // Continue fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    
    return 0; // keep the input
}