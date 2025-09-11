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
        
        // Get a dimension to unsqueeze
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply unsqueeze_copy operation
        torch::Tensor result = torch::unsqueeze_copy(input_tensor, dim);
        
        // Verify the result has one more dimension than the input
        if (result.dim() != input_tensor.dim() + 1) {
            throw std::runtime_error("Unexpected dimension count after unsqueeze_copy");
        }
        
        // Try another dimension if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t dim2;
            std::memcpy(&dim2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Apply unsqueeze_copy again on the result
            torch::Tensor result2 = torch::unsqueeze_copy(result, dim2);
            
            // Verify the result has one more dimension than before
            if (result2.dim() != result.dim() + 1) {
                throw std::runtime_error("Unexpected dimension count after second unsqueeze_copy");
            }
        }
        
        // Try unsqueeze_copy on a scalar tensor if we have a tensor with no dimensions
        if (input_tensor.dim() == 0) {
            torch::Tensor scalar_result = torch::unsqueeze_copy(input_tensor, 0);
            if (scalar_result.dim() != 1) {
                throw std::runtime_error("Unexpected dimension count after unsqueeze_copy on scalar");
            }
        }
        
        // Try unsqueeze_copy with a negative dimension
        if (offset + sizeof(int64_t) <= Size) {
            int64_t neg_dim;
            std::memcpy(&neg_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure it's negative
            neg_dim = -std::abs(neg_dim) - 1;
            
            // Apply unsqueeze_copy with negative dimension
            torch::Tensor neg_result = torch::unsqueeze_copy(input_tensor, neg_dim);
            
            // Verify the result has one more dimension than the input
            if (neg_result.dim() != input_tensor.dim() + 1) {
                throw std::runtime_error("Unexpected dimension count after unsqueeze_copy with negative dim");
            }
        }
        
        // Try unsqueeze_copy with a very large dimension (should throw)
        if (offset + sizeof(int64_t) <= Size) {
            int64_t large_dim;
            std::memcpy(&large_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make it large but not too large to avoid undefined behavior
            large_dim = std::abs(large_dim) + input_tensor.dim() + 10;
            
            try {
                torch::Tensor large_result = torch::unsqueeze_copy(input_tensor, large_dim);
            } catch (const c10::Error &e) {
                // Expected exception for out-of-bounds dimension
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
