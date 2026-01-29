#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dim parameter from the remaining data
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure dim is within valid range for the tensor
        if (input_tensor.dim() > 0) {
            dim = dim % input_tensor.dim();
            if (dim < 0) {
                dim += input_tensor.dim();
            }
        } else {
            dim = 0;
        }
        
        // Apply cumprod operation
        torch::Tensor result = torch::cumprod(input_tensor, dim);
        
        // Try with optional dtype parameter
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                torch::Tensor result_with_dtype = torch::cumprod(input_tensor, dim, dtype);
            } catch (const std::exception &) {
                // dtype conversion may fail for some combinations, expected
            }
        }
        
        // Try with out parameter
        if (input_tensor.dim() > 0) {
            try {
                torch::Tensor out_tensor = torch::empty_like(input_tensor);
                torch::cumprod_out(out_tensor, input_tensor, dim);
            } catch (const std::exception &) {
                // out tensor shape mismatch can occur, expected
            }
        }
        
        // Try the method version
        torch::Tensor result_method = input_tensor.cumprod(dim);
        
        // Try with negative dim
        if (input_tensor.dim() > 0) {
            int64_t neg_dim = -1;
            torch::Tensor result_neg_dim = torch::cumprod(input_tensor, neg_dim);
        }
        
        // Try with edge case dimensions
        if (input_tensor.dim() > 0) {
            // Dimension at the boundary
            int64_t boundary_dim = input_tensor.dim() - 1;
            torch::Tensor result_boundary = torch::cumprod(input_tensor, boundary_dim);
        }
        
        // Test on 0-dim (scalar) tensor
        if (offset < Size) {
            torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset]));
            torch::Tensor scalar_result = torch::cumprod(scalar_tensor, 0);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}