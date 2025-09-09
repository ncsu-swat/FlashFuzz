#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor shape and data type
        auto shape = parse_tensor_shape(Data, Size, offset);
        if (shape.empty()) return 0;
        
        auto dtype = parse_dtype(Data, Size, offset);
        
        // Create input tensor
        auto input = create_tensor(Data, Size, offset, shape, dtype);
        if (!input.defined()) return 0;
        
        // Parse dimensions for amax operation
        auto dims = parse_dims(Data, Size, offset, input.dim());
        
        // Parse keepdim flag
        bool keepdim = parse_bool(Data, Size, offset);
        
        // Test torch.amax with different parameter combinations
        
        // Case 1: amax without any parameters (reduce all dimensions)
        auto result1 = torch::amax(input);
        
        // Case 2: amax with specific dimensions
        if (!dims.empty()) {
            auto result2 = torch::amax(input, dims);
            
            // Case 3: amax with dimensions and keepdim
            auto result3 = torch::amax(input, dims, keepdim);
        }
        
        // Case 4: amax with single dimension
        if (input.dim() > 0) {
            int64_t single_dim = parse_int64(Data, Size, offset) % input.dim();
            auto result4 = torch::amax(input, single_dim);
            auto result5 = torch::amax(input, single_dim, keepdim);
        }
        
        // Test edge cases with empty dimensions
        std::vector<int64_t> empty_dims;
        auto result6 = torch::amax(input, empty_dims, keepdim);
        
        // Test with all dimensions
        std::vector<int64_t> all_dims;
        for (int64_t i = 0; i < input.dim(); ++i) {
            all_dims.push_back(i);
        }
        if (!all_dims.empty()) {
            auto result7 = torch::amax(input, all_dims, keepdim);
        }
        
        // Test with negative dimension indices
        if (input.dim() > 0) {
            std::vector<int64_t> neg_dims;
            for (int64_t i = 0; i < std::min(static_cast<int64_t>(3), input.dim()); ++i) {
                neg_dims.push_back(-1 - i);
            }
            auto result8 = torch::amax(input, neg_dims, keepdim);
        }
        
        // Test with duplicate dimensions (should handle gracefully or throw)
        if (input.dim() > 1) {
            std::vector<int64_t> dup_dims = {0, 0, 1};
            try {
                auto result9 = torch::amax(input, dup_dims, keepdim);
            } catch (...) {
                // Expected to potentially throw for duplicate dimensions
            }
        }
        
        // Test with out-of-bounds dimensions (should throw)
        if (input.dim() > 0) {
            try {
                std::vector<int64_t> oob_dims = {input.dim()};
                auto result10 = torch::amax(input, oob_dims, keepdim);
            } catch (...) {
                // Expected to throw for out-of-bounds dimensions
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}