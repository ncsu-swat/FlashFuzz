#include "fuzzer_utils.h"
#include <iostream>
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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, create a tensor with same shape but different values
            input2 = torch::ones_like(input1);
        }
        
        // Apply torch.fmin operation
        // fmin returns the element-wise minimum, propagating NaN
        torch::Tensor result = torch::fmin(input1, input2);
        
        // Try scalar version
        if (offset + sizeof(float) <= Size) {
            float scalar_value;
            memcpy(&scalar_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            try {
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor scalar_result1 = torch::fmin(input1, scalar_tensor);
                torch::Tensor scalar_result2 = torch::fmin(scalar_tensor, input1);
            } catch (const std::exception&) {
                // Some scalar operations might fail depending on tensor type
            }
        }
        
        // Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_result = torch::fmin(empty_tensor, empty_tensor);
        } catch (const std::exception&) {
            // Empty tensor operations might fail
        }
        
        // Test with special float values if input is float type
        if (input1.is_floating_point()) {
            try {
                // Test NaN propagation behavior
                torch::Tensor nan_tensor = torch::full_like(input1, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor nan_result = torch::fmin(input1, nan_tensor);
            } catch (const std::exception&) {
                // NaN operations might behave differently for some dtypes
            }
            
            try {
                // Test infinity handling
                torch::Tensor inf_tensor = torch::full_like(input1, std::numeric_limits<float>::infinity());
                torch::Tensor inf_result = torch::fmin(input1, inf_tensor);
                
                torch::Tensor neg_inf_tensor = torch::full_like(input1, -std::numeric_limits<float>::infinity());
                torch::Tensor neg_inf_result = torch::fmin(input1, neg_inf_tensor);
            } catch (const std::exception&) {
                // Infinity operations might fail for some dtypes
            }
        }
        
        // Test with same tensor (should return identical tensor)
        try {
            torch::Tensor same_result = torch::fmin(input1, input1);
        } catch (const std::exception&) {
            // Might fail for unsupported dtypes
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}