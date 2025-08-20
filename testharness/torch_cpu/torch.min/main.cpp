#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test different variants of torch::min
        
        // 1. Global min - returns a single value tensor
        torch::Tensor global_min = torch::min(input);
        
        // 2. Min along dimension with keepdim=false (default)
        if (offset < Size && input.dim() > 0) {
            int64_t dim = static_cast<int64_t>(Data[offset++]) % input.dim();
            torch::Tensor values, indices;
            std::tie(values, indices) = torch::min(input, dim);
        }
        
        // 3. Min along dimension with keepdim=true
        if (offset < Size && input.dim() > 0) {
            int64_t dim = static_cast<int64_t>(Data[offset++]) % input.dim();
            bool keepdim = offset < Size && (Data[offset++] % 2 == 0);
            torch::Tensor values, indices;
            std::tie(values, indices) = torch::min(input, dim, keepdim);
        }
        
        // 4. Element-wise min of two tensors
        if (offset + 1 < Size) {
            torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try element-wise min even if shapes don't match (should throw if incompatible)
            torch::Tensor elementwise_min = torch::min(input, other);
        }
        
        // 5. Min with named dimension (if tensor has names)
        if (offset < Size && input.dim() > 0) {
            // Try to create a named tensor
            try {
                std::vector<torch::Dimname> names;
                for (int64_t i = 0; i < input.dim(); i++) {
                    names.push_back(torch::Dimname::wildcard());
                }
                torch::Tensor named_input = input.refine_names(names);
                
                // Get min along a named dimension
                torch::Tensor values, indices;
                std::tie(values, indices) = torch::min(named_input, names[0]);
            } catch (...) {
                // Ignore errors with named tensors
            }
        }
        
        // 6. Test with empty tensor
        if (offset < Size) {
            try {
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_tensor = torch::empty(empty_shape, input.options());
                torch::Tensor empty_min = torch::min(empty_tensor);
            } catch (...) {
                // Ignore errors with empty tensors
            }
        }
        
        // 7. Test with scalar tensor
        if (offset < Size) {
            try {
                torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset++]));
                torch::Tensor scalar_min = torch::min(scalar_tensor);
            } catch (...) {
                // Ignore errors with scalar tensors
            }
        }
        
        // 8. Test with negative dimension
        if (offset < Size && input.dim() > 0) {
            try {
                int64_t neg_dim = -1 * (static_cast<int64_t>(Data[offset++]) % input.dim() + 1);
                torch::Tensor values, indices;
                std::tie(values, indices) = torch::min(input, neg_dim);
            } catch (...) {
                // Ignore errors with negative dimensions
            }
        }
        
        // 9. Test with out-of-bounds dimension
        if (offset < Size) {
            try {
                int64_t out_of_bounds_dim = input.dim() + (Data[offset++] % 5 + 1);
                torch::Tensor values, indices;
                std::tie(values, indices) = torch::min(input, out_of_bounds_dim);
            } catch (...) {
                // Ignore errors with out-of-bounds dimensions
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