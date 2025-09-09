#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation and axis parameters
        if (Size < 16) return 0;

        // Generate tensor dimensions (1-6 dimensions)
        auto num_dims = consume_integral_in_range<int>(Data, Size, offset, 1, 6);
        std::vector<int64_t> dims;
        for (int i = 0; i < num_dims; ++i) {
            dims.push_back(consume_integral_in_range<int64_t>(Data, Size, offset, 1, 10));
        }

        // Create input tensor with random data
        auto input = torch::randn(dims);

        // Test single axis moveaxis
        if (offset + 4 <= Size) {
            auto source = consume_integral_in_range<int>(Data, Size, offset, -num_dims, num_dims - 1);
            auto destination = consume_integral_in_range<int>(Data, Size, offset, -num_dims, num_dims - 1);
            
            auto result1 = torch::moveaxis(input, source, destination);
            
            // Verify result has same number of elements
            if (result1.numel() != input.numel()) {
                std::cerr << "moveaxis changed number of elements" << std::endl;
            }
        }

        // Test multiple axes moveaxis with vectors
        if (offset + 8 <= Size && num_dims >= 2) {
            auto num_axes = consume_integral_in_range<int>(Data, Size, offset, 1, std::min(num_dims, 4));
            
            std::vector<int64_t> source_axes;
            std::vector<int64_t> dest_axes;
            
            for (int i = 0; i < num_axes; ++i) {
                source_axes.push_back(consume_integral_in_range<int>(Data, Size, offset, -num_dims, num_dims - 1));
                dest_axes.push_back(consume_integral_in_range<int>(Data, Size, offset, -num_dims, num_dims - 1));
            }
            
            auto result2 = torch::moveaxis(input, source_axes, dest_axes);
            
            // Verify result has same number of elements
            if (result2.numel() != input.numel()) {
                std::cerr << "moveaxis with vectors changed number of elements" << std::endl;
            }
        }

        // Test edge cases with different tensor types
        if (offset + 2 <= Size) {
            auto dtype_choice = consume_integral_in_range<int>(Data, Size, offset, 0, 3);
            torch::Tensor typed_input;
            
            switch (dtype_choice) {
                case 0: typed_input = input.to(torch::kFloat32); break;
                case 1: typed_input = input.to(torch::kFloat64); break;
                case 2: typed_input = input.to(torch::kInt32); break;
                case 3: typed_input = input.to(torch::kInt64); break;
            }
            
            if (offset + 4 <= Size) {
                auto source = consume_integral_in_range<int>(Data, Size, offset, 0, num_dims - 1);
                auto destination = consume_integral_in_range<int>(Data, Size, offset, 0, num_dims - 1);
                
                auto result3 = torch::moveaxis(typed_input, source, destination);
                
                // Verify dtype is preserved
                if (result3.dtype() != typed_input.dtype()) {
                    std::cerr << "moveaxis changed tensor dtype" << std::endl;
                }
            }
        }

        // Test with empty tensor
        if (dims.size() > 0) {
            auto empty_dims = dims;
            empty_dims[0] = 0; // Make first dimension empty
            auto empty_tensor = torch::empty(empty_dims);
            
            if (offset + 4 <= Size) {
                auto source = consume_integral_in_range<int>(Data, Size, offset, -num_dims, num_dims - 1);
                auto destination = consume_integral_in_range<int>(Data, Size, offset, -num_dims, num_dims - 1);
                
                auto result4 = torch::moveaxis(empty_tensor, source, destination);
                
                // Should still work with empty tensors
                if (result4.numel() != 0) {
                    std::cerr << "moveaxis with empty tensor produced non-empty result" << std::endl;
                }
            }
        }

        // Test with 1D tensor (edge case)
        if (offset + 2 <= Size) {
            auto tensor_1d = torch::randn({5});
            auto result5 = torch::moveaxis(tensor_1d, 0, 0); // Should be no-op
            
            if (!torch::equal(tensor_1d, result5)) {
                std::cerr << "moveaxis 1D tensor self-move failed" << std::endl;
            }
        }

        // Test negative indexing
        if (offset + 4 <= Size && num_dims >= 2) {
            auto source = consume_integral_in_range<int>(Data, Size, offset, -num_dims, -1);
            auto destination = consume_integral_in_range<int>(Data, Size, offset, -num_dims, -1);
            
            auto result6 = torch::moveaxis(input, source, destination);
            
            // Verify result dimensions
            if (result6.dim() != input.dim()) {
                std::cerr << "moveaxis changed number of dimensions" << std::endl;
            }
        }

        // Test with contiguous and non-contiguous tensors
        if (num_dims >= 2 && offset + 4 <= Size) {
            auto non_contiguous = input.transpose(0, 1);
            auto source = consume_integral_in_range<int>(Data, Size, offset, 0, num_dims - 1);
            auto destination = consume_integral_in_range<int>(Data, Size, offset, 0, num_dims - 1);
            
            auto result7 = torch::moveaxis(non_contiguous, source, destination);
            
            // Should work regardless of contiguity
            if (result7.numel() != non_contiguous.numel()) {
                std::cerr << "moveaxis failed with non-contiguous tensor" << std::endl;
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