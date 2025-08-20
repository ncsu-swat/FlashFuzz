#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor for indices
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are integers
        if (!indices.dtype().is_integral_type()) {
            indices = indices.to(torch::kInt64);
        }
        
        // Parse dimensions
        std::vector<int64_t> dims;
        const size_t max_dims = 5;
        
        // Determine number of dimensions to use
        uint8_t num_dims = 1;
        if (offset < Size) {
            num_dims = (Data[offset++] % max_dims) + 1;
        }
        
        // Parse each dimension
        for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; ++i) {
            int64_t dim_value;
            std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure dimension is positive (but allow zero for edge case testing)
            if (dim_value < 0) {
                dim_value = -dim_value;
            }
            
            // Limit dimension size to avoid excessive memory usage
            dim_value = dim_value % 1000;
            
            dims.push_back(dim_value);
        }
        
        // If we couldn't parse any dimensions, use a default
        if (dims.empty()) {
            dims.push_back(10);
        }
        
        // Try different variants of unravel_index
        
        // Variant 1: Using a scalar index
        if (indices.numel() == 1 || indices.dim() == 0) {
            auto result1 = at::unravel_index(indices, dims);
        }
        
        // Variant 2: Using a tensor of indices
        if (indices.numel() > 0) {
            // Reshape indices to 1D if needed
            if (indices.dim() > 1) {
                indices = indices.reshape(-1);
            }
            
            auto result2 = at::unravel_index(indices, dims);
        }
        
        // Variant 3: Using a list of dimensions
        if (indices.numel() > 0) {
            auto result3 = at::unravel_index(indices, c10::IntArrayRef(dims));
        }
        
        // Variant 4: Edge case with empty dimensions
        if (offset < Size && Data[offset] % 5 == 0) {
            std::vector<int64_t> empty_dims;
            try {
                auto result4 = at::unravel_index(indices, empty_dims);
            } catch (...) {
                // Expected to throw, continue
            }
        }
        
        // Variant 5: Edge case with zero dimensions
        if (offset < Size && Data[offset] % 5 == 1) {
            std::vector<int64_t> zero_dims = {0};
            try {
                auto result5 = at::unravel_index(indices, zero_dims);
            } catch (...) {
                // Expected to throw, continue
            }
        }
        
        // Variant 6: Edge case with negative indices
        if (offset < Size && Data[offset] % 5 == 2) {
            torch::Tensor neg_indices = -indices.abs();
            try {
                auto result6 = at::unravel_index(neg_indices, dims);
            } catch (...) {
                // Expected to throw, continue
            }
        }
        
        // Variant 7: Edge case with indices larger than product of dimensions
        if (offset < Size && Data[offset] % 5 == 3) {
            int64_t prod = 1;
            for (auto d : dims) {
                if (d > 0 && prod < std::numeric_limits<int64_t>::max() / d) {
                    prod *= d;
                }
            }
            
            if (prod > 0) {
                torch::Tensor large_indices = indices.abs() + prod;
                try {
                    auto result7 = at::unravel_index(large_indices, dims);
                } catch (...) {
                    // Expected to throw, continue
                }
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