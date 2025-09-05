#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>
#include <limits>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal data for one tensor and clip parameters
        if (Size < 4) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse min/max values for clipping
        torch::Scalar min_val;
        torch::Scalar max_val;
        bool has_min = false;
        bool has_max = false;
        
        // Check if we have data for min/max flags
        if (offset < Size) {
            uint8_t flags = Data[offset++];
            has_min = flags & 0x01;
            has_max = flags & 0x02;
        }
        
        // Parse min value if flag is set and data available
        if (has_min && offset + sizeof(double) <= Size) {
            double min_double;
            std::memcpy(&min_double, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Handle special values
            if (std::isnan(min_double) || std::isinf(min_double)) {
                min_val = min_double;
            } else {
                // Bound the value to reasonable range
                min_val = std::max(-1e10, std::min(1e10, min_double));
            }
        }
        
        // Parse max value if flag is set and data available
        if (has_max && offset + sizeof(double) <= Size) {
            double max_double;
            std::memcpy(&max_double, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Handle special values
            if (std::isnan(max_double) || std::isinf(max_double)) {
                max_val = max_double;
            } else {
                // Bound the value to reasonable range
                max_val = std::max(-1e10, std::min(1e10, max_double));
            }
        }
        
        // Test various clip configurations
        torch::Tensor result;
        
        // Case 1: clip with both min and max
        if (has_min && has_max) {
            try {
                result = torch::clip(input, min_val, max_val);
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors (e.g., invalid min/max combination)
                // Continue execution
            }
        }
        
        // Case 2: clip with only min
        if (has_min && !has_max) {
            try {
                result = torch::clip(input, min_val, c10::nullopt);
            } catch (const c10::Error& e) {
                // Continue execution
            }
        }
        
        // Case 3: clip with only max
        if (!has_min && has_max) {
            try {
                result = torch::clip(input, c10::nullopt, max_val);
            } catch (const c10::Error& e) {
                // Continue execution
            }
        }
        
        // Case 4: clip with neither (should be identity)
        if (!has_min && !has_max) {
            try {
                result = torch::clip(input, c10::nullopt, c10::nullopt);
            } catch (const c10::Error& e) {
                // Continue execution
            }
        }
        
        // Test edge cases with tensor min/max values
        if (offset < Size) {
            uint8_t tensor_test = Data[offset++];
            
            // Test with tensor-based min/max
            if (tensor_test & 0x01) {
                try {
                    // Create min tensor with same shape or broadcastable shape
                    torch::Tensor min_tensor;
                    if (tensor_test & 0x02) {
                        // Same shape as input
                        min_tensor = torch::randn_like(input);
                    } else {
                        // Scalar tensor
                        min_tensor = torch::tensor(has_min ? min_val.toDouble() : -1.0, input.options());
                    }
                    
                    result = torch::clip(input, min_tensor, c10::nullopt);
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
            
            if (tensor_test & 0x04) {
                try {
                    // Create max tensor
                    torch::Tensor max_tensor;
                    if (tensor_test & 0x08) {
                        // Same shape as input
                        max_tensor = torch::randn_like(input);
                    } else {
                        // Scalar tensor
                        max_tensor = torch::tensor(has_max ? max_val.toDouble() : 1.0, input.options());
                    }
                    
                    result = torch::clip(input, c10::nullopt, max_tensor);
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
            
            // Test with both tensor min and max
            if (tensor_test & 0x10) {
                try {
                    torch::Tensor min_tensor = torch::randn_like(input) - 1.0;
                    torch::Tensor max_tensor = torch::randn_like(input) + 1.0;
                    result = torch::clip(input, min_tensor, max_tensor);
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
        }
        
        // Test in-place operation if we have more data
        if (offset < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag & 0x01) {
                try {
                    torch::Tensor input_copy = input.clone();
                    input_copy.clip_(has_min ? min_val : c10::nullopt, 
                                     has_max ? max_val : c10::nullopt);
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
        }
        
        // Test with different tensor configurations
        if (offset < Size) {
            uint8_t config = Data[offset++];
            
            // Test with non-contiguous tensor
            if (config & 0x01 && input.numel() > 1) {
                try {
                    torch::Tensor transposed = input.dim() > 1 ? input.transpose(0, -1) : input;
                    result = torch::clip(transposed, has_min ? min_val : c10::nullopt,
                                       has_max ? max_val : c10::nullopt);
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
            
            // Test with view
            if (config & 0x02 && input.numel() > 0) {
                try {
                    torch::Tensor viewed = input.view({-1});
                    result = torch::clip(viewed, has_min ? min_val : c10::nullopt,
                                       has_max ? max_val : c10::nullopt);
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
            
            // Test with requires_grad
            if (config & 0x04 && input.dtype() == torch::kFloat) {
                try {
                    torch::Tensor grad_input = input.requires_grad_(true);
                    result = torch::clip(grad_input, has_min ? min_val : c10::nullopt,
                                       has_max ? max_val : c10::nullopt);
                    if (result.requires_grad()) {
                        // Trigger backward pass
                        result.sum().backward();
                    }
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
        }
        
        // Test special floating point values if input is floating point
        if (input.is_floating_point() && offset < Size) {
            uint8_t special_test = Data[offset++];
            
            if (special_test & 0x01) {
                try {
                    // Test with NaN min/max
                    result = torch::clip(input, std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN());
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
            
            if (special_test & 0x02) {
                try {
                    // Test with infinity
                    result = torch::clip(input, -std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity());
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
            
            if (special_test & 0x04) {
                try {
                    // Test with inverted min/max (max < min)
                    result = torch::clip(input, 1.0, -1.0);
                } catch (const c10::Error& e) {
                    // Continue execution
                }
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}