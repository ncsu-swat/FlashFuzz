#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// Helper to consume a scalar value from fuzzer data
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T default_val) {
    if (offset + sizeof(T) > size) {
        return default_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

// Helper to get a scalar based on dtype
torch::Scalar getScalarForDtype(torch::ScalarType dtype, const uint8_t* data, size_t& offset, size_t size) {
    switch (dtype) {
        case torch::kFloat:
        case torch::kFloat32:
            return torch::Scalar(consumeValue<float>(data, offset, size, 0.0f));
        case torch::kDouble:
        case torch::kFloat64:
            return torch::Scalar(consumeValue<double>(data, offset, size, 0.0));
        case torch::kHalf:
            return torch::Scalar(consumeValue<float>(data, offset, size, 0.0f));
        case torch::kBFloat16:
            return torch::Scalar(consumeValue<float>(data, offset, size, 0.0f));
        case torch::kInt8:
            return torch::Scalar(consumeValue<int8_t>(data, offset, size, 0));
        case torch::kUInt8:
            return torch::Scalar(consumeValue<uint8_t>(data, offset, size, 0));
        case torch::kInt16:
            return torch::Scalar(consumeValue<int16_t>(data, offset, size, 0));
        case torch::kInt32:
            return torch::Scalar(consumeValue<int32_t>(data, offset, size, 0));
        case torch::kInt64:
            return torch::Scalar(consumeValue<int64_t>(data, offset, size, 0));
        case torch::kBool:
            return torch::Scalar(consumeValue<uint8_t>(data, offset, size, 0) != 0);
        case torch::kComplexFloat:
        case torch::kComplexDouble:
            // For complex types, use real part only for min value
            return torch::Scalar(consumeValue<float>(data, offset, size, 0.0f));
        default:
            return torch::Scalar(0.0f);
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check for basic operations
        if (Size < 4) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with minimal tensor
            input_tensor = torch::randn({1});
        }

        // Get the dtype of the input tensor for consistent min value
        torch::ScalarType input_dtype = input_tensor.scalar_type();
        
        // Parse min value - make it compatible with tensor dtype
        torch::Scalar min_val = getScalarForDtype(input_dtype, Data, offset, Size);
        
        // Test 1: Basic clamp_min operation
        try {
            torch::Tensor result = torch::clamp_min(input_tensor, min_val);
            
            // Verify output shape matches input
            if (result.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch after clamp_min!" << std::endl;
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are acceptable for invalid operations
        }

        // Test 2: In-place clamp_min operation
        if (offset < Size) {
            try {
                torch::Tensor tensor_copy = input_tensor.clone();
                tensor_copy.clamp_min_(min_val);
            } catch (const c10::Error& e) {
                // In-place operation might fail for certain dtypes
            }
        }

        // Test 3: Test with tensor min value (if we have more data)
        if (offset + 2 < Size) {
            try {
                torch::Tensor min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Broadcasting test - clamp_min with tensor
                torch::Tensor result = torch::clamp_min(input_tensor, min_tensor);
            } catch (const std::exception& e) {
                // Broadcasting might fail, that's ok
            }
        }

        // Test 4: Edge cases with special values
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                // Test with infinity
                if (input_tensor.is_floating_point()) {
                    torch::Tensor result_inf = torch::clamp_min(input_tensor, 
                        std::numeric_limits<float>::infinity());
                    
                    // Test with negative infinity
                    torch::Tensor result_neginf = torch::clamp_min(input_tensor, 
                        -std::numeric_limits<float>::infinity());
                    
                    // Test with NaN
                    torch::Tensor result_nan = torch::clamp_min(input_tensor, 
                        std::numeric_limits<float>::quiet_NaN());
                }
            } catch (const c10::Error& e) {
                // Special values might cause issues
            }
        }

        // Test 5: Test with different tensor layouts
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                // Test with non-contiguous tensor
                if (input_tensor.dim() > 1) {
                    torch::Tensor transposed = input_tensor.transpose(0, -1);
                    torch::Tensor result = torch::clamp_min(transposed, min_val);
                }
                
                // Test with view
                if (input_tensor.numel() > 1) {
                    torch::Tensor viewed = input_tensor.view({-1});
                    torch::Tensor result = torch::clamp_min(viewed, min_val);
                }
            } catch (const c10::Error& e) {
                // View/transpose operations might fail
            }
        }

        // Test 6: Test with requires_grad
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
                    torch::Tensor grad_tensor = input_tensor.clone().requires_grad_(true);
                    torch::Tensor result = torch::clamp_min(grad_tensor, min_val);
                    
                    // Test backward pass
                    if (result.requires_grad()) {
                        torch::Tensor grad_out = torch::ones_like(result);
                        result.backward(grad_out);
                    }
                }
            } catch (const c10::Error& e) {
                // Autograd might fail for certain operations
            }
        }

        // Test 7: Test with empty and zero-dimensional tensors
        if (offset < Size && Data[offset++] % 5 == 0) {
            try {
                // Empty tensor
                torch::Tensor empty = torch::empty({0});
                torch::Tensor result_empty = torch::clamp_min(empty, min_val);
                
                // Scalar tensor
                torch::Tensor scalar = torch::tensor(3.14);
                torch::Tensor result_scalar = torch::clamp_min(scalar, min_val);
            } catch (const c10::Error& e) {
                // Edge cases might fail
            }
        }

        // Test 8: Test output parameter variant
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::Tensor out = torch::empty_like(input_tensor);
                torch::clamp_min_out(out, input_tensor, min_val);
            } catch (const c10::Error& e) {
                // Output variant might have different constraints
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