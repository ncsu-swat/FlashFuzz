#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some bytes for tensor creation and alpha parameter
        if (Size < 3) {
            // Not enough data to create meaningful input
            return 0;
        }

        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse alpha parameter from remaining bytes
        double alpha = 1.0; // default value
        if (offset < Size) {
            // Use remaining bytes to determine alpha
            size_t remaining = Size - offset;
            if (remaining >= sizeof(double)) {
                std::memcpy(&alpha, Data + offset, sizeof(double));
                offset += sizeof(double);
            } else if (remaining >= sizeof(float)) {
                float alpha_f;
                std::memcpy(&alpha_f, Data + offset, sizeof(float));
                alpha = static_cast<double>(alpha_f);
                offset += sizeof(float);
            } else {
                // Use single byte scaled to reasonable range
                uint8_t alpha_byte = Data[offset++];
                // Map to range [-10, 10] with some special cases
                if (alpha_byte == 0) {
                    alpha = 0.0;
                } else if (alpha_byte == 255) {
                    alpha = std::numeric_limits<double>::infinity();
                } else if (alpha_byte == 254) {
                    alpha = -std::numeric_limits<double>::infinity();
                } else if (alpha_byte == 253) {
                    alpha = std::numeric_limits<double>::quiet_NaN();
                } else {
                    alpha = (static_cast<double>(alpha_byte) / 127.5) * 10.0 - 10.0;
                }
            }
        }

        // Clone the input tensor to preserve original for comparison if needed
        torch::Tensor input_clone = input.clone();
        
        // Test different tensor configurations
        // 1. Test with contiguous tensor
        if (input.is_contiguous()) {
            try {
                input.celu_(alpha);
            } catch (const c10::Error& e) {
                // PyTorch-specific errors - these are expected for invalid inputs
                // Continue execution
            }
        }
        
        // 2. Test with non-contiguous tensor (if we have enough data)
        if (offset < Size && Size - offset > 0) {
            uint8_t permute_flag = Data[offset++];
            if (permute_flag % 2 == 0 && input.dim() >= 2) {
                // Create a non-contiguous view by transposing
                torch::Tensor transposed = input_clone.transpose(0, input_clone.dim() - 1);
                try {
                    transposed.celu_(alpha);
                } catch (const c10::Error& e) {
                    // Expected for some configurations
                }
            }
        }
        
        // 3. Test with different memory layouts if tensor has multiple dimensions
        if (input_clone.dim() > 1 && offset < Size) {
            uint8_t layout_selector = Data[offset++];
            
            // Try different stride patterns
            if (layout_selector % 3 == 0) {
                // Create a sliced view
                torch::Tensor sliced = input_clone.slice(0, 0, input_clone.size(0), 2);
                if (sliced.numel() > 0) {
                    try {
                        sliced.celu_(alpha);
                    } catch (const c10::Error& e) {
                        // Expected for some configurations
                    }
                }
            } else if (layout_selector % 3 == 1) {
                // Create a view with different shape if possible
                if (input_clone.numel() > 1 && input_clone.numel() % 2 == 0) {
                    torch::Tensor reshaped = input_clone.view({-1, 2});
                    try {
                        reshaped.celu_(alpha);
                    } catch (const c10::Error& e) {
                        // Expected for some configurations
                    }
                }
            }
        }
        
        // 4. Test with zero-dimensional tensor (scalar)
        if (offset < Size && Data[offset++] % 4 == 0) {
            torch::Tensor scalar_tensor = torch::tensor(3.14, input.options());
            try {
                scalar_tensor.celu_(alpha);
            } catch (const c10::Error& e) {
                // Expected for some configurations
            }
        }
        
        // 5. Test with empty tensor
        if (offset < Size && Data[offset++] % 5 == 0) {
            torch::Tensor empty_tensor = torch::empty({0}, input.options());
            try {
                empty_tensor.celu_(alpha);
            } catch (const c10::Error& e) {
                // Expected for some configurations
            }
        }
        
        // 6. Test with different data types if we have more data
        if (offset < Size) {
            uint8_t dtype_test = Data[offset++];
            
            // Test with integer types (which might not be supported)
            if (dtype_test % 3 == 0 && input.numel() > 0) {
                torch::Tensor int_tensor = input_clone.to(torch::kInt32);
                try {
                    int_tensor.celu_(alpha);
                } catch (const c10::Error& e) {
                    // Expected - CELU might not support integer types
                }
            }
            
            // Test with complex types
            if (dtype_test % 3 == 1 && input.numel() > 0) {
                torch::Tensor complex_tensor = input_clone.to(torch::kComplexFloat);
                try {
                    complex_tensor.celu_(alpha);
                } catch (const c10::Error& e) {
                    // Expected - CELU might not support complex types
                }
            }
        }
        
        // 7. Test requires_grad scenarios
        if (offset < Size && Data[offset++] % 2 == 0) {
            if (input_clone.dtype() == torch::kFloat || 
                input_clone.dtype() == torch::kDouble ||
                input_clone.dtype() == torch::kHalf ||
                input_clone.dtype() == torch::kBFloat16) {
                torch::Tensor grad_tensor = input_clone.clone().requires_grad_(true);
                try {
                    grad_tensor.celu_(alpha);
                    // Try backward pass if successful
                    if (grad_tensor.requires_grad() && grad_tensor.numel() > 0) {
                        torch::Tensor grad_output = torch::ones_like(grad_tensor);
                        grad_tensor.backward(grad_output);
                    }
                } catch (const c10::Error& e) {
                    // Expected for some configurations
                }
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
        return -1; // discard the input
    }
    
    return 0; // keep the input
}