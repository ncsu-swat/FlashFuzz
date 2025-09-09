#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor dimensions and axis values
        if (Size < 16) {
            return 0;
        }

        // Generate tensor dimensions (1-6 dimensions)
        int num_dims = (Data[offset] % 6) + 1;
        offset++;

        std::vector<int64_t> dims;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            // Keep dimensions reasonable to avoid memory issues
            int64_t dim_size = (Data[offset] % 10) + 1;
            dims.push_back(dim_size);
            offset++;
        }

        if (dims.empty()) {
            dims.push_back(1); // Fallback to 1D tensor
        }

        // Create input tensor with random data
        torch::Tensor input = torch::randn(dims);

        // Test different data types
        if (offset < Size) {
            int dtype_choice = Data[offset] % 6;
            offset++;
            
            switch (dtype_choice) {
                case 0: input = input.to(torch::kFloat32); break;
                case 1: input = input.to(torch::kFloat64); break;
                case 2: input = input.to(torch::kInt32); break;
                case 3: input = input.to(torch::kInt64); break;
                case 4: input = input.to(torch::kBool); break;
                case 5: input = input.to(torch::kInt8); break;
            }
        }

        // Generate axis0 and axis1 values
        int64_t axis0, axis1;
        
        if (offset + 1 < Size) {
            // Test both positive and negative axis values
            axis0 = static_cast<int64_t>(static_cast<int8_t>(Data[offset])) % static_cast<int64_t>(dims.size());
            offset++;
            axis1 = static_cast<int64_t>(static_cast<int8_t>(Data[offset])) % static_cast<int64_t>(dims.size());
            offset++;
        } else {
            // Fallback values
            axis0 = 0;
            axis1 = dims.size() > 1 ? 1 : 0;
        }

        // Test torch::swapaxes with the generated parameters
        torch::Tensor result = torch::swapaxes(input, axis0, axis1);

        // Verify the result has the expected properties
        if (result.defined()) {
            // Check that dimensions are preserved (just swapped)
            auto result_sizes = result.sizes();
            auto input_sizes = input.sizes();
            
            if (result_sizes.size() == input_sizes.size()) {
                // Verify the swap worked correctly
                int64_t normalized_axis0 = axis0 < 0 ? axis0 + input.dim() : axis0;
                int64_t normalized_axis1 = axis1 < 0 ? axis1 + input.dim() : axis1;
                
                if (normalized_axis0 >= 0 && normalized_axis0 < input.dim() &&
                    normalized_axis1 >= 0 && normalized_axis1 < input.dim()) {
                    
                    // Additional operations to stress test the result
                    auto sum_result = result.sum();
                    auto mean_result = result.mean();
                    
                    // Test chaining swapaxes operations
                    if (offset < Size && dims.size() > 2) {
                        int64_t axis2 = Data[offset] % dims.size();
                        auto chained = torch::swapaxes(result, normalized_axis0, axis2);
                    }
                }
            }
        }

        // Test edge cases with same axis values
        if (offset < Size) {
            torch::Tensor same_axis_result = torch::swapaxes(input, axis0, axis0);
            // Should be equivalent to the original tensor
        }

        // Test with extreme axis values to trigger error handling
        if (offset < Size) {
            try {
                int64_t large_axis = 1000 + (Data[offset] % 100);
                torch::swapaxes(input, 0, large_axis);
            } catch (...) {
                // Expected to fail for out-of-bounds axes
            }
        }

        // Test with very negative axis values
        if (offset < Size) {
            try {
                int64_t very_negative_axis = -1000 - (Data[offset] % 100);
                torch::swapaxes(input, very_negative_axis, 0);
            } catch (...) {
                // Expected to fail for out-of-bounds axes
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