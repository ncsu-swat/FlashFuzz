#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation and permutation
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to get dimensions from
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get the number of dimensions
        int64_t ndim = tensor.dim();
        
        // Create permutation vector
        std::vector<int64_t> permutation;
        
        // Generate permutation based on remaining data
        for (int64_t i = 0; i < ndim && offset < Size; ++i) {
            uint8_t perm_value = Data[offset++];
            permutation.push_back(perm_value % ndim);
        }
        
        // If we don't have enough data for a full permutation, use identity permutation
        if (permutation.size() < static_cast<size_t>(ndim)) {
            permutation.clear();
            for (int64_t i = 0; i < ndim; ++i) {
                permutation.push_back(i);
            }
        }
        
        // Get the shape of the tensor
        std::vector<int64_t> shape;
        for (int64_t i = 0; i < ndim; ++i) {
            shape.push_back(tensor.size(i));
        }
        
        // Create permuted shape
        std::vector<int64_t> permuted_shape;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i < static_cast<int64_t>(permutation.size())) {
                int64_t perm_idx = permutation[i];
                if (perm_idx >= 0 && perm_idx < ndim) {
                    permuted_shape.push_back(shape[perm_idx]);
                } else {
                    permuted_shape.push_back(shape[i % ndim]);
                }
            } else {
                permuted_shape.push_back(shape[i]);
            }
        }
        
        // Call empty_permuted with the permutation
        torch::Tensor result = torch::empty_permuted(shape, permutation, tensor.options());
        
        // Verify the result has the expected shape
        if (result.dim() != ndim) {
            throw std::runtime_error("Result tensor has unexpected number of dimensions");
        }
        
        // Try to access elements of the tensor to ensure it's valid
        if (result.numel() > 0) {
            result.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}