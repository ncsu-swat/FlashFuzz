#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract min and max values for clipping if we have enough data
        float min_val = -1.0f;
        float max_val = 1.0f;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize NaN/Inf values
            if (std::isnan(min_val) || std::isinf(min_val)) {
                min_val = -1.0f;
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize NaN/Inf values
            if (std::isnan(max_val) || std::isinf(max_val)) {
                max_val = 1.0f;
            }
        }
        
        // Swap if min > max to ensure valid range
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // 1. Test clip_ with both min and max using Scalar values
        {
            auto tensor_copy = tensor.clone();
            tensor_copy.clip_(min_val, max_val);
        }
        
        // 2. Test clip_ with only min (max = nullopt)
        {
            auto tensor_copy = tensor.clone();
            tensor_copy.clip_(min_val, c10::nullopt);
        }
        
        // 3. Test clip_ with only max (min = nullopt)
        {
            auto tensor_copy = tensor.clone();
            tensor_copy.clip_(c10::nullopt, max_val);
        }
        
        // 4. Test clip_ with Tensor min/max
        if (offset + 1 <= Size) {
            bool use_scalar_tensors = Data[offset++] & 0x1;
            
            torch::Tensor min_tensor, max_tensor;
            
            if (use_scalar_tensors) {
                min_tensor = torch::tensor(min_val);
                max_tensor = torch::tensor(max_val);
            } else {
                min_tensor = torch::full_like(tensor, min_val);
                max_tensor = torch::full_like(tensor, max_val);
            }
            
            // Test with both tensor min and max
            {
                auto tensor_copy = tensor.clone();
                tensor_copy.clip_(min_tensor, max_tensor);
            }
            
            // Test with only tensor min
            {
                auto tensor_copy = tensor.clone();
                tensor_copy.clip_(min_tensor, c10::nullopt);
            }
            
            // Test with only tensor max
            {
                auto tensor_copy = tensor.clone();
                tensor_copy.clip_(c10::nullopt, max_tensor);
            }
        }
        
        // 5. Edge case: test with min == max (clamping to a single value)
        if (offset + sizeof(float) <= Size) {
            float same_val = 0.0f;
            std::memcpy(&same_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Sanitize
            if (std::isnan(same_val) || std::isinf(same_val)) {
                same_val = 0.0f;
            }
            
            auto tensor_copy = tensor.clone();
            tensor_copy.clip_(same_val, same_val);
        }
        
        // 6. Test with negative range
        {
            auto tensor_copy = tensor.clone();
            tensor_copy.clip_(-10.0f, -1.0f);
        }
        
        // 7. Test with zero crossing range
        {
            auto tensor_copy = tensor.clone();
            tensor_copy.clip_(-5.0f, 5.0f);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}