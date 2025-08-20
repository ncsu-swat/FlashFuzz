#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dimension to unbind along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply unbind_copy operation
        std::vector<torch::Tensor> result;
        
        // Try different variants of unbind_copy
        if (input_tensor.dim() > 0) {
            // Apply unbind_copy with explicit dimension
            result = torch::unbind_copy(input_tensor, dim);
            
            // Verify the result
            if (input_tensor.dim() > 0) {
                int64_t expected_size = input_tensor.size(dim % input_tensor.dim());
                if (result.size() != expected_size) {
                    throw std::runtime_error("Unexpected result size from unbind_copy");
                }
            }
            
            // Try to access and perform operations on the resulting tensors
            for (const auto& tensor : result) {
                // Simple operation to ensure tensor is valid
                auto sum = tensor.sum();
            }
        }
        
        // Try unbind_copy with default dimension (0)
        if (input_tensor.dim() > 0) {
            auto result_default = torch::unbind_copy(input_tensor);
            
            // Verify the result
            int64_t expected_size = input_tensor.size(0);
            if (result_default.size() != expected_size) {
                throw std::runtime_error("Unexpected result size from default unbind_copy");
            }
            
            // Try to access and perform operations on the resulting tensors
            for (const auto& tensor : result_default) {
                // Simple operation to ensure tensor is valid
                auto sum = tensor.sum();
            }
        }
        
        // Try unbind_copy on a scalar tensor (should throw an exception)
        if (input_tensor.dim() == 0) {
            try {
                auto result_scalar = torch::unbind_copy(input_tensor);
            } catch (const c10::Error& e) {
                // Expected exception for scalar tensor
            }
        }
        
        // Try unbind_copy with out-of-bounds dimension
        if (input_tensor.dim() > 0) {
            try {
                int64_t out_of_bounds_dim = input_tensor.dim() + std::abs(dim % 10);
                auto result_oob = torch::unbind_copy(input_tensor, out_of_bounds_dim);
            } catch (const c10::Error& e) {
                // Expected exception for out-of-bounds dimension
            }
            
            try {
                int64_t negative_dim = -input_tensor.dim() - 1 - std::abs(dim % 10);
                auto result_neg = torch::unbind_copy(input_tensor, negative_dim);
            } catch (const c10::Error& e) {
                // Expected exception for negative out-of-bounds dimension
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