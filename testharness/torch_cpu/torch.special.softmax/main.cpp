#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Softmax requires floating point tensors
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Get a dimension to apply softmax along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // If tensor has dimensions, use modulo to get a valid dimension
        if (input_tensor.dim() > 0) {
            dim = dim % input_tensor.dim();
        } else {
            dim = 0;
        }
        
        // Apply softmax operation
        torch::Tensor result = torch::special::softmax(input_tensor, dim, std::nullopt);
        
        // Try with optional dtype parameter - only use valid floating point types
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t dtype_selector = Data[offset];
            offset += sizeof(uint8_t);
            
            // Only use valid floating point dtypes for softmax output
            torch::ScalarType dtype;
            switch (dtype_selector % 4) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kBFloat16; break;
                case 3: dtype = torch::kFloat; break; // fallback to float
            }
            
            try {
                torch::Tensor result_with_dtype = torch::special::softmax(input_tensor, dim, dtype);
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }
        
        // Try with double precision
        try {
            torch::Tensor double_tensor = input_tensor.to(torch::kDouble);
            torch::Tensor double_result = torch::special::softmax(double_tensor, dim, std::nullopt);
        } catch (...) {
            // Silently ignore conversion failures
        }
        
        // Try with different dimensions if tensor has multiple dimensions
        if (input_tensor.dim() > 1) {
            for (int64_t alt_dim = 0; alt_dim < input_tensor.dim(); alt_dim++) {
                if (alt_dim != dim) {
                    try {
                        torch::Tensor alt_result = torch::special::softmax(input_tensor, alt_dim, std::nullopt);
                    } catch (...) {
                        // Silently ignore
                    }
                    break; // Just try one alternative dimension
                }
            }
        }
        
        // Try with negative dimension
        if (input_tensor.dim() > 0) {
            int64_t neg_dim = -1;
            try {
                torch::Tensor neg_dim_result = torch::special::softmax(input_tensor, neg_dim, std::nullopt);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Try with a contiguous tensor
        if (!input_tensor.is_contiguous()) {
            try {
                torch::Tensor contig_tensor = input_tensor.contiguous();
                torch::Tensor contig_result = torch::special::softmax(contig_tensor, dim, std::nullopt);
            } catch (...) {
                // Silently ignore
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