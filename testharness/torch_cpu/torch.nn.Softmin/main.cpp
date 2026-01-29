#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty or scalar tensors for meaningful softmin
        if (input.numel() == 0 || input.dim() == 0) {
            return 0;
        }
        
        // Parse dimension parameter from the remaining data
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte;
            std::memcpy(&dim_byte, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            // Normalize dimension to valid range
            dim = dim_byte % input.dim();
            // Handle negative dimensions (Python-style)
            if (dim < 0) {
                dim += input.dim();
            }
        }
        
        // Test Softmin module
        try {
            torch::nn::Softmin softmin_module{torch::nn::SoftminOptions{dim}};
            torch::Tensor output = softmin_module->forward(input);
            
            // Force computation
            (void)output.sum().item<float>();
        } catch (const std::exception &) {
            // Silently catch shape/dimension mismatches
        }
        
        // Test functional interface
        try {
            torch::Tensor output_functional = torch::nn::functional::softmin(
                input, 
                torch::nn::functional::SoftminFuncOptions{dim}
            );
            
            // Force computation
            (void)output_functional.sum().item<float>();
        } catch (const std::exception &) {
            // Silently catch shape/dimension mismatches
        }
        
        // Try with different dimension if there's more data
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim2_byte;
            std::memcpy(&dim2_byte, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            int64_t dim2 = dim2_byte % input.dim();
            if (dim2 < 0) {
                dim2 += input.dim();
            }
            
            try {
                torch::nn::Softmin softmin_module2{torch::nn::SoftminOptions{dim2}};
                torch::Tensor output2 = softmin_module2->forward(input);
                (void)output2.sum().item<float>();
            } catch (const std::exception &) {
                // Silently catch
            }
        }
        
        // Test with last dimension (common use case)
        try {
            int64_t last_dim = input.dim() - 1;
            torch::nn::Softmin last_dim_softmin{torch::nn::SoftminOptions{last_dim}};
            torch::Tensor last_dim_output = last_dim_softmin->forward(input);
            (void)last_dim_output.sum().item<float>();
        } catch (const std::exception &) {
            // Silently catch
        }
        
        // Test with negative dimension index
        try {
            torch::nn::Softmin neg_dim_softmin{torch::nn::SoftminOptions{-1}};
            torch::Tensor neg_dim_output = neg_dim_softmin->forward(input);
            (void)neg_dim_output.sum().item<float>();
        } catch (const std::exception &) {
            // Silently catch
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}