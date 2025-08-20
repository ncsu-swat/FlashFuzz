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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to use for cummin
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, make sure dim is within valid range
            if (input.dim() > 0) {
                dim = dim % input.dim();
                if (dim < 0) {
                    dim += input.dim();
                }
            }
        }
        
        // Apply cummin operation
        std::tuple<torch::Tensor, torch::Tensor> result = torch::cummin(input, dim);
        
        // Access the values and indices from the result
        torch::Tensor values = std::get<0>(result);
        torch::Tensor indices = std::get<1>(result);
        
        // Verify the output tensors have the same shape as the input
        if (values.sizes() != input.sizes() || indices.sizes() != input.sizes()) {
            throw std::runtime_error("Output tensor shapes don't match input tensor shape");
        }
        
        // Try a different variant of cummin with named parameters
        auto result2 = torch::cummin(input, dim);
        
        // Try cummin with a negative dimension
        if (input.dim() > 0) {
            int64_t neg_dim = -1;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&neg_dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Make sure it's a valid negative dimension
                neg_dim = -(std::abs(neg_dim) % input.dim());
                if (neg_dim == 0) neg_dim = -1;
                
                auto result_neg = torch::cummin(input, neg_dim);
            }
        }
        
        // Try cummin on a zero-sized tensor if we have one
        if (input.numel() == 0 && input.dim() > 0) {
            auto zero_result = torch::cummin(input, dim);
        }
        
        // Try cummin on a scalar tensor
        if (input.dim() == 0) {
            try {
                auto scalar_result = torch::cummin(input, 0);
            } catch (const c10::Error& e) {
                // Expected exception for scalar tensor
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