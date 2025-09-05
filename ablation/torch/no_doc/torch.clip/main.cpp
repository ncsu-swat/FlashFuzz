#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// Helper to consume a value of type T from fuzzer data
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T default_val = T{}) {
    if (offset + sizeof(T) > size) {
        return default_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

// Helper to create a scalar from fuzzer data with the same dtype as tensor
torch::Scalar createScalarFromData(const uint8_t* data, size_t& offset, size_t size, torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat:
            return torch::Scalar(consumeValue<float>(data, offset, size, 0.0f));
        case torch::kDouble:
            return torch::Scalar(consumeValue<double>(data, offset, size, 0.0));
        case torch::kHalf:
            return torch::Scalar(static_cast<float>(consumeValue<uint16_t>(data, offset, size, 0)));
        case torch::kBFloat16:
            return torch::Scalar(static_cast<float>(consumeValue<uint16_t>(data, offset, size, 0)));
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
            // For complex types, just use real part for clipping bounds
            return torch::Scalar(consumeValue<float>(data, offset, size, 0.0f));
        default:
            return torch::Scalar(0);
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    if (Size < 4) {
        return 0; // Need minimum bytes for operation
    }

    try
    {
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Consume operation mode byte
        uint8_t op_mode = consumeValue<uint8_t>(Data, offset, Size, 0);
        
        // Determine clipping strategy based on op_mode
        uint8_t clip_variant = op_mode % 8;
        
        torch::Tensor result;
        
        switch (clip_variant) {
            case 0: {
                // clip with both min and max as scalars
                torch::Scalar min_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                torch::Scalar max_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                result = torch::clip(input_tensor, min_val, max_val);
                break;
            }
            case 1: {
                // clip with only min as scalar
                torch::Scalar min_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                result = torch::clip(input_tensor, min_val, c10::nullopt);
                break;
            }
            case 2: {
                // clip with only max as scalar
                torch::Scalar max_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                result = torch::clip(input_tensor, c10::nullopt, max_val);
                break;
            }
            case 3: {
                // clip with min and max as tensors (broadcasting test)
                torch::Tensor min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor max_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Convert to same dtype as input if needed
                if (min_tensor.scalar_type() != input_tensor.scalar_type()) {
                    min_tensor = min_tensor.to(input_tensor.scalar_type());
                }
                if (max_tensor.scalar_type() != input_tensor.scalar_type()) {
                    max_tensor = max_tensor.to(input_tensor.scalar_type());
                }
                
                result = torch::clip(input_tensor, min_tensor, max_tensor);
                break;
            }
            case 4: {
                // clip with min as tensor, max as scalar
                torch::Tensor min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Scalar max_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                
                if (min_tensor.scalar_type() != input_tensor.scalar_type()) {
                    min_tensor = min_tensor.to(input_tensor.scalar_type());
                }
                
                result = torch::clip(input_tensor, min_tensor, max_val);
                break;
            }
            case 5: {
                // clip with min as scalar, max as tensor
                torch::Scalar min_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                torch::Tensor max_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                if (max_tensor.scalar_type() != input_tensor.scalar_type()) {
                    max_tensor = max_tensor.to(input_tensor.scalar_type());
                }
                
                result = torch::clip(input_tensor, min_val, max_tensor);
                break;
            }
            case 6: {
                // clip with special values (inf, -inf, nan)
                uint8_t special_case = consumeValue<uint8_t>(Data, offset, Size, 0) % 6;
                torch::Scalar min_val, max_val;
                
                switch (special_case) {
                    case 0:
                        min_val = torch::Scalar(-std::numeric_limits<float>::infinity());
                        max_val = torch::Scalar(std::numeric_limits<float>::infinity());
                        break;
                    case 1:
                        min_val = torch::Scalar(std::numeric_limits<float>::quiet_NaN());
                        max_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                        break;
                    case 2:
                        min_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                        max_val = torch::Scalar(std::numeric_limits<float>::quiet_NaN());
                        break;
                    case 3:
                        min_val = torch::Scalar(0);
                        max_val = torch::Scalar(-1); // max < min
                        break;
                    case 4:
                        min_val = torch::Scalar(std::numeric_limits<float>::max());
                        max_val = torch::Scalar(std::numeric_limits<float>::min());
                        break;
                    default:
                        min_val = torch::Scalar(-std::numeric_limits<float>::epsilon());
                        max_val = torch::Scalar(std::numeric_limits<float>::epsilon());
                        break;
                }
                
                result = torch::clip(input_tensor, min_val, max_val);
                break;
            }
            case 7: {
                // In-place clipping
                torch::Scalar min_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                torch::Scalar max_val = createScalarFromData(Data, offset, Size, input_tensor.scalar_type());
                
                // Make a copy for in-place operation
                torch::Tensor tensor_copy = input_tensor.clone();
                tensor_copy.clip_(min_val, max_val);
                result = tensor_copy;
                break;
            }
        }
        
        // Additional operations to increase coverage
        uint8_t post_op = consumeValue<uint8_t>(Data, offset, Size, 0) % 4;
        
        switch (post_op) {
            case 0:
                // Check if result is finite (for floating point tensors)
                if (result.is_floating_point()) {
                    auto is_finite = torch::isfinite(result);
                }
                break;
            case 1:
                // Check for NaN values
                if (result.is_floating_point()) {
                    auto has_nan = torch::isnan(result).any();
                }
                break;
            case 2:
                // Compute statistics on result
                if (result.numel() > 0) {
                    auto min_elem = result.min();
                    auto max_elem = result.max();
                    if (result.is_floating_point()) {
                        auto mean_val = result.mean();
                    }
                }
                break;
            case 3:
                // Test gradient computation if applicable
                if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
                    torch::Tensor grad_input = input_tensor.clone().requires_grad_(true);
                    torch::Scalar min_val = createScalarFromData(Data, offset, Size, grad_input.scalar_type());
                    torch::Scalar max_val = createScalarFromData(Data, offset, Size, grad_input.scalar_type());
                    auto grad_result = torch::clip(grad_input, min_val, max_val);
                    if (grad_result.numel() > 0) {
                        grad_result.sum().backward();
                    }
                }
                break;
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
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}