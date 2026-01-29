#include "fuzzer_utils.h"
#include <iostream>

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
        // Need sufficient data for tensor creation and parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract batch size (1-8) and dim choice from fuzzer data
        uint8_t batch_param = Data[offset++] % 8 + 1;
        uint8_t dim_choice = Data[offset++];
        
        // Cross product requires vectors of size 3 in the specified dimension
        // Create tensors with shape that guarantees a dimension of size 3
        
        // Variant based on fuzzer input
        int variant = dim_choice % 5;
        
        try {
            torch::Tensor input1, input2;
            int64_t dim = -1;
            
            switch (variant) {
                case 0: {
                    // Simple 1D vectors of size 3
                    input1 = torch::randn({3});
                    input2 = torch::randn({3});
                    // dim defaults to -1 (last dimension)
                    torch::Tensor result = torch::cross(input1, input2);
                    break;
                }
                case 1: {
                    // 2D tensors with last dim = 3
                    int64_t batch = static_cast<int64_t>(batch_param);
                    input1 = torch::randn({batch, 3});
                    input2 = torch::randn({batch, 3});
                    dim = 1;
                    torch::Tensor result = torch::cross(input1, input2, dim);
                    break;
                }
                case 2: {
                    // 2D tensors with first dim = 3
                    int64_t batch = static_cast<int64_t>(batch_param);
                    input1 = torch::randn({3, batch});
                    input2 = torch::randn({3, batch});
                    dim = 0;
                    torch::Tensor result = torch::cross(input1, input2, dim);
                    break;
                }
                case 3: {
                    // 3D tensors
                    int64_t batch1 = static_cast<int64_t>((batch_param % 4) + 1);
                    int64_t batch2 = static_cast<int64_t>((batch_param / 4) + 1);
                    input1 = torch::randn({batch1, batch2, 3});
                    input2 = torch::randn({batch1, batch2, 3});
                    dim = 2;
                    torch::Tensor result = torch::cross(input1, input2, dim);
                    break;
                }
                case 4: {
                    // Broadcasting case: different batch dimensions
                    int64_t batch = static_cast<int64_t>(batch_param);
                    input1 = torch::randn({batch, 1, 3});
                    input2 = torch::randn({1, batch, 3});
                    dim = -1;
                    torch::Tensor result = torch::cross(input1, input2, dim);
                    break;
                }
            }
        } catch (...) {
            // Silently ignore expected failures from shape/type mismatches
        }
        
        // Test with different dtypes
        try {
            torch::Tensor input1 = torch::randn({3}, torch::kFloat64);
            torch::Tensor input2 = torch::randn({3}, torch::kFloat64);
            torch::Tensor result = torch::cross(input1, input2);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with complex tensors
        try {
            torch::Tensor input1 = torch::randn({3}, torch::kComplexFloat);
            torch::Tensor input2 = torch::randn({3}, torch::kComplexFloat);
            torch::Tensor result = torch::cross(input1, input2);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with tensors created from fuzzer data
        if (offset + 24 <= Size) {
            try {
                // Create 3-element vectors from fuzzer data
                float data1[3], data2[3];
                std::memcpy(data1, Data + offset, sizeof(float) * 3);
                offset += sizeof(float) * 3;
                std::memcpy(data2, Data + offset, sizeof(float) * 3);
                offset += sizeof(float) * 3;
                
                torch::Tensor input1 = torch::from_blob(data1, {3}, torch::kFloat).clone();
                torch::Tensor input2 = torch::from_blob(data2, {3}, torch::kFloat).clone();
                torch::Tensor result = torch::cross(input1, input2);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Test with requires_grad for autograd coverage
        try {
            torch::Tensor input1 = torch::randn({3}, torch::requires_grad());
            torch::Tensor input2 = torch::randn({3}, torch::requires_grad());
            torch::Tensor result = torch::cross(input1, input2);
            result.sum().backward();
        } catch (...) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}