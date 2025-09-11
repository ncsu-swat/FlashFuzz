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
        
        // Parse permutation dimensions from remaining data
        std::vector<int64_t> permutation;
        int64_t rank = input_tensor.dim();
        
        // Generate permutation dimensions
        for (int64_t i = 0; i < rank && offset < Size; ++i) {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t dim_value;
                std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Ensure dimension is within valid range [0, rank-1]
                int64_t dim = std::abs(dim_value) % rank;
                
                // Check if this dimension is already in the permutation
                if (std::find(permutation.begin(), permutation.end(), dim) == permutation.end()) {
                    permutation.push_back(dim);
                }
            }
        }
        
        // If we didn't get a complete permutation, fill in the missing dimensions
        if (permutation.size() < static_cast<size_t>(rank)) {
            std::vector<bool> used(rank, false);
            for (int64_t dim : permutation) {
                used[dim] = true;
            }
            
            for (int64_t i = 0; i < rank; ++i) {
                if (!used[i]) {
                    permutation.push_back(i);
                }
            }
        }
        
        // Apply permute operation (permute_copy doesn't exist, use permute)
        torch::Tensor output;
        
        // Handle edge cases
        if (rank == 0) {
            // Scalar tensor - permutation is empty
            output = input_tensor.clone();
        } else {
            output = input_tensor.permute(permutation);
        }
        
        // Verify the output tensor has the expected shape
        if (rank > 0) {
            std::vector<int64_t> expected_shape;
            for (int64_t dim : permutation) {
                expected_shape.push_back(input_tensor.size(dim));
            }
            
            auto output_sizes = output.sizes();
            for (size_t i = 0; i < expected_shape.size(); ++i) {
                if (i < output_sizes.size() && expected_shape[i] != output_sizes[i]) {
                    throw std::runtime_error("Output shape mismatch");
                }
            }
        }
        
        // Test that the data is correctly permuted
        // For small tensors, we can check some values
        if (input_tensor.numel() > 0 && input_tensor.numel() < 1000) {
            // Access some elements to ensure no crashes
            output[0].item();
            
            // For rank > 1 tensors, check a few more indices
            if (rank > 1 && output.numel() > 1) {
                std::vector<at::indexing::TensorIndex> indices;
                for (int64_t i = 0; i < rank; ++i) {
                    indices.push_back(0);
                }
                // Set last index to 1 if possible
                if (output.size(rank-1) > 1) {
                    indices[rank-1] = 1;
                }
                output.index(indices).item();
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
