#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dimension parameter for log_softmax
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            // Normalize dim to valid range based on tensor dimensions
            if (input.dim() > 0) {
                dim = raw_dim % input.dim();
                // Handle negative dims
                if (dim < 0) {
                    dim += input.dim();
                }
            }
        }
        
        // Apply log_softmax operation
        try {
            torch::Tensor result = torch::log_softmax(input, dim);
        } catch (const std::exception &e) {
            // Expected failures for invalid shapes/dims - catch silently
        }
        
        // Try another variant with optional dtype parameter
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                // Apply log_softmax with dtype (must be floating point)
                torch::Tensor result_with_dtype = torch::log_softmax(input, dim, dtype);
            } catch (const std::exception &e) {
                // Expected failures for incompatible dtypes - catch silently
            }
        }
        
        // Try functional variant with LogSoftmaxFuncOptions
        try {
            auto options = torch::nn::functional::LogSoftmaxFuncOptions(dim);
            torch::Tensor result_functional = torch::nn::functional::log_softmax(input, options);
        } catch (const std::exception &e) {
            // Expected failures - catch silently
        }
        
        // Try with negative dimension indexing
        if (input.dim() > 0) {
            try {
                int64_t neg_dim = -1;
                torch::Tensor result_neg_dim = torch::log_softmax(input, neg_dim);
            } catch (const std::exception &e) {
                // Expected failures - catch silently
            }
        }
        
        // Try with different tensor types
        try {
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor result_float = torch::log_softmax(float_input, dim);
        } catch (const std::exception &e) {
            // Expected failures - catch silently
        }
        
        try {
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor result_double = torch::log_softmax(double_input, dim);
        } catch (const std::exception &e) {
            // Expected failures - catch silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}