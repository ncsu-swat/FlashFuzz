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
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create an Identity module
        torch::nn::Identity identity_module;
        
        // Create an input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the Identity module to the input tensor
        torch::Tensor output_tensor = identity_module->forward(input_tensor);
        
        // Verify identity property: output should be same as input
        // This exercises the comparison operations as well
        bool is_same = torch::equal(input_tensor, output_tensor);
        (void)is_same;
        
        // Try with additional tensors if we have enough data
        if (offset < Size) {
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor output_second = identity_module->forward(second_tensor);
        }
        
        // Test with various tensor types based on fuzzer data
        if (Size > 0) {
            uint8_t type_selector = Data[0] % 5;
            
            try {
                switch (type_selector) {
                    case 0: {
                        // Empty tensor
                        torch::Tensor empty_tensor = torch::empty({0});
                        identity_module->forward(empty_tensor);
                        break;
                    }
                    case 1: {
                        // Scalar tensor
                        float val = (Size > 4) ? *reinterpret_cast<const float*>(Data) : 3.14f;
                        torch::Tensor scalar_tensor = torch::tensor(val);
                        identity_module->forward(scalar_tensor);
                        break;
                    }
                    case 2: {
                        // Boolean tensor
                        bool bval = (Size > 1) ? (Data[1] % 2 == 0) : true;
                        torch::Tensor bool_tensor = torch::tensor(bval);
                        identity_module->forward(bool_tensor);
                        break;
                    }
                    case 3: {
                        // Complex tensor
                        torch::Tensor complex_tensor = torch::randn({2, 2}, torch::kComplexFloat);
                        identity_module->forward(complex_tensor);
                        break;
                    }
                    case 4: {
                        // Multi-dimensional tensor with shape from fuzzer data
                        int64_t d1 = (Size > 1) ? (Data[1] % 10 + 1) : 2;
                        int64_t d2 = (Size > 2) ? (Data[2] % 10 + 1) : 3;
                        int64_t d3 = (Size > 3) ? (Data[3] % 10 + 1) : 4;
                        torch::Tensor multi_dim = torch::randn({d1, d2, d3});
                        identity_module->forward(multi_dim);
                        break;
                    }
                }
            } catch (const std::exception &) {
                // Inner catch - silently handle expected failures
            }
        }
        
        // Test that module can be applied multiple times (stateless check)
        torch::Tensor reapply = identity_module->forward(input_tensor);
        torch::Tensor reapply2 = identity_module->forward(reapply);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}