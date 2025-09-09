#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation and dimension parameters
        if (Size < 16) {
            return 0;
        }

        // Generate tensor dimensions (1-6 dimensions to cover various cases)
        uint8_t num_dims = (Data[offset++] % 6) + 1;
        std::vector<int64_t> dims;
        
        for (int i = 0; i < num_dims && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 10) + 1; // 1-10 size per dimension
            dims.push_back(dim_size);
        }

        if (offset >= Size) return 0;

        // Create input tensor with random data
        torch::Tensor input = torch::randn(dims);

        // Generate source and destination dimension indices
        uint8_t num_moves = (Data[offset++] % std::min(num_dims, 4)) + 1; // 1-4 moves max
        
        std::vector<int64_t> source_dims;
        std::vector<int64_t> dest_dims;

        for (int i = 0; i < num_moves && offset + 1 < Size; i++) {
            // Source dimension (can be negative)
            int64_t src = static_cast<int64_t>(static_cast<int8_t>(Data[offset++])) % num_dims;
            // Destination dimension (can be negative)  
            int64_t dst = static_cast<int64_t>(static_cast<int8_t>(Data[offset++])) % num_dims;
            
            source_dims.push_back(src);
            dest_dims.push_back(dst);
        }

        if (source_dims.empty()) return 0;

        // Test torch::movedim with vector arguments
        torch::Tensor result1 = torch::movedim(input, source_dims, dest_dims);

        // Test single dimension move if we have at least one pair
        if (!source_dims.empty()) {
            torch::Tensor result2 = torch::movedim(input, source_dims[0], dest_dims[0]);
        }

        // Test edge cases with different tensor types
        if (offset < Size) {
            uint8_t tensor_type = Data[offset++] % 4;
            torch::Tensor typed_input;
            
            switch (tensor_type) {
                case 0:
                    typed_input = input.to(torch::kFloat32);
                    break;
                case 1:
                    typed_input = input.to(torch::kFloat64);
                    break;
                case 2:
                    typed_input = input.to(torch::kInt32);
                    break;
                case 3:
                    typed_input = input.to(torch::kInt64);
                    break;
            }
            
            torch::Tensor typed_result = torch::movedim(typed_input, source_dims, dest_dims);
        }

        // Test with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        if (empty_tensor.dim() > 0) {
            try {
                torch::Tensor empty_result = torch::movedim(empty_tensor, {0}, {0});
            } catch (...) {
                // Expected to potentially fail
            }
        }

        // Test with scalar tensor (0-dimensional)
        torch::Tensor scalar = torch::tensor(42.0);
        try {
            torch::Tensor scalar_result = torch::movedim(scalar, {}, {});
        } catch (...) {
            // May fail for scalar tensors
        }

        // Test boundary conditions - maximum negative indices
        if (num_dims > 1) {
            try {
                torch::Tensor boundary_result = torch::movedim(input, -num_dims, num_dims - 1);
            } catch (...) {
                // May fail with out of bounds
            }
        }

        // Test with duplicate source dimensions (should fail)
        if (num_dims > 2) {
            try {
                std::vector<int64_t> dup_source = {0, 0};
                std::vector<int64_t> dup_dest = {1, 2};
                torch::Tensor dup_result = torch::movedim(input, dup_source, dup_dest);
            } catch (...) {
                // Expected to fail with duplicate source dims
            }
        }

        // Test with mismatched source/dest vector sizes
        if (source_dims.size() > 1) {
            try {
                std::vector<int64_t> short_dest = {dest_dims[0]};
                torch::Tensor mismatch_result = torch::movedim(input, source_dims, short_dest);
            } catch (...) {
                // Expected to fail with size mismatch
            }
        }

        // Test with very large tensor if we have remaining data
        if (offset < Size && Size > 50) {
            try {
                std::vector<int64_t> large_dims = {100, 50};
                torch::Tensor large_tensor = torch::zeros(large_dims);
                torch::Tensor large_result = torch::movedim(large_tensor, 0, 1);
            } catch (...) {
                // May fail due to memory constraints
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