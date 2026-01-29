#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a floating point tensor for softmax
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Get dimension for softmax from fuzzer data
        int64_t dim = 0;
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t dim_byte = Data[offset++];
            // Map to valid dimension range
            if (input_tensor.dim() > 0) {
                dim = static_cast<int64_t>(dim_byte % input_tensor.dim());
            }
        }
        
        // Create Softmax module with proper options (use brace initialization to avoid vexing parse)
        torch::nn::Softmax softmax_module{torch::nn::SoftmaxOptions(dim)};
        
        // Apply softmax operation
        torch::Tensor output = softmax_module->forward(input_tensor);
        
        // Try with different valid dimensions
        if (input_tensor.dim() > 0 && offset < Size) {
            uint8_t dim2_byte = Data[offset++];
            int64_t dim2 = static_cast<int64_t>(dim2_byte % input_tensor.dim());
            
            torch::nn::Softmax softmax_module2{torch::nn::SoftmaxOptions(dim2)};
            torch::Tensor output2 = softmax_module2->forward(input_tensor);
        }
        
        // Try with last dimension (common use case)
        if (input_tensor.dim() > 0) {
            torch::nn::Softmax last_dim_softmax{torch::nn::SoftmaxOptions(-1)};
            torch::Tensor last_dim_output = last_dim_softmax->forward(input_tensor);
        }
        
        // Try with negative dimension indexing
        if (input_tensor.dim() > 1 && offset < Size) {
            uint8_t neg_dim_byte = Data[offset++];
            int64_t neg_dim = -static_cast<int64_t>((neg_dim_byte % input_tensor.dim()) + 1);
            
            torch::nn::Softmax neg_softmax{torch::nn::SoftmaxOptions(neg_dim)};
            torch::Tensor neg_output = neg_softmax->forward(input_tensor);
        }
        
        // Try with different dtypes
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 3;
            
            try {
                torch::Tensor converted_input;
                switch (dtype_choice) {
                    case 0:
                        converted_input = input_tensor.to(torch::kFloat);
                        break;
                    case 1:
                        converted_input = input_tensor.to(torch::kDouble);
                        break;
                    case 2:
                        converted_input = input_tensor.to(torch::kHalf);
                        break;
                    default:
                        converted_input = input_tensor;
                }
                
                if (converted_input.dim() > 0) {
                    torch::nn::Softmax dtype_softmax{torch::nn::SoftmaxOptions(dim)};
                    torch::Tensor dtype_output = dtype_softmax->forward(converted_input);
                }
            } catch (const std::exception &) {
                // Silently ignore dtype conversion failures
            }
        }
        
        // Test with a multi-dimensional tensor created from remaining data
        if (offset + 4 <= Size && input_tensor.dim() > 0) {
            try {
                // Create a batch of inputs
                torch::Tensor batched = input_tensor.unsqueeze(0).expand({2, -1});
                for (int i = 0; i < batched.dim(); i++) {
                    torch::nn::Softmax batch_softmax{torch::nn::SoftmaxOptions(i)};
                    torch::Tensor batch_output = batch_softmax->forward(batched);
                }
            } catch (const std::exception &) {
                // Silently ignore shape-related failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}