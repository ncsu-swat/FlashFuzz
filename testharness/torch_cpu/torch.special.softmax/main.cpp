#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to apply softmax along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // If tensor has dimensions, use modulo to get a valid dimension
        if (input_tensor.dim() > 0) {
            dim = dim % input_tensor.dim();
        }
        
        // Apply softmax operation
        torch::Tensor result = torch::special::softmax(input_tensor, dim, std::nullopt);
        
        // Try with optional dtype parameter if we have more data
        if (offset + sizeof(int) <= Size) {
            int dtype_val;
            std::memcpy(&dtype_val, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // Use modulo to get a valid dtype
            torch::ScalarType dtype = static_cast<torch::ScalarType>(abs(dtype_val) % 12);
            
            torch::Tensor result_with_dtype = torch::special::softmax(input_tensor, dim, dtype);
        }
        
        // Try with half precision if we have a floating point tensor
        if (input_tensor.is_floating_point()) {
            torch::Tensor half_tensor = input_tensor.to(torch::kHalf);
            torch::Tensor half_result = torch::special::softmax(half_tensor, dim, std::nullopt);
        }
        
        // Try with different dimensions if tensor has multiple dimensions
        if (input_tensor.dim() > 1) {
            for (int64_t alt_dim = 0; alt_dim < input_tensor.dim(); alt_dim++) {
                if (alt_dim != dim) {
                    torch::Tensor alt_result = torch::special::softmax(input_tensor, alt_dim, std::nullopt);
                    break; // Just try one alternative dimension to avoid excessive computation
                }
            }
        }
        
        // Try with negative dimension
        if (input_tensor.dim() > 0) {
            int64_t neg_dim = -1;
            torch::Tensor neg_dim_result = torch::special::softmax(input_tensor, neg_dim, std::nullopt);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
