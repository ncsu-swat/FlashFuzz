#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor dimensions and parameters
        if (Size < 16) {
            return 0;
        }

        // Extract number of dimensions (1-6 to keep it reasonable)
        int num_dims = (Data[offset] % 6) + 1;
        offset++;

        // Extract tensor dimensions
        std::vector<int64_t> dims;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            // Keep dimensions reasonable (1-10) to avoid memory issues
            int64_t dim = (Data[offset] % 10) + 1;
            dims.push_back(dim);
            offset++;
        }

        if (dims.empty()) {
            dims.push_back(1); // Fallback to 1D tensor
        }

        // Calculate total elements needed
        int64_t total_elements = 1;
        for (auto dim : dims) {
            total_elements *= dim;
        }

        // Limit total elements to prevent excessive memory usage
        if (total_elements > 10000) {
            return 0;
        }

        // Create tensor with random data
        torch::Tensor input = torch::randn(dims);

        // Test case 1: Default flatten (start_dim=0, end_dim=-1)
        torch::Tensor result1 = torch::flatten(input);

        // Extract start_dim and end_dim from fuzzer data
        if (offset + 1 < Size) {
            // Map to valid dimension range
            int start_dim_raw = static_cast<int>(Data[offset]);
            int end_dim_raw = static_cast<int>(Data[offset + 1]);
            offset += 2;

            // Convert to valid dimension indices
            int max_dim = static_cast<int>(dims.size()) - 1;
            int start_dim = (start_dim_raw % (2 * dims.size())) - static_cast<int>(dims.size());
            int end_dim = (end_dim_raw % (2 * dims.size())) - static_cast<int>(dims.size());

            // Test case 2: Custom start_dim and end_dim
            torch::Tensor result2 = torch::flatten(input, start_dim, end_dim);

            // Test case 3: start_dim only (end_dim defaults to -1)
            torch::Tensor result3 = torch::flatten(input, start_dim);

            // Test edge cases
            if (offset < Size) {
                // Test case 4: Same start and end dim
                int same_dim = (Data[offset] % dims.size());
                torch::Tensor result4 = torch::flatten(input, same_dim, same_dim);
                offset++;
            }

            // Test case 5: Negative indices
            if (offset < Size) {
                int neg_start = -((Data[offset] % dims.size()) + 1);
                torch::Tensor result5 = torch::flatten(input, neg_start);
                offset++;
            }
        }

        // Test with different tensor types if we have more data
        if (offset < Size) {
            torch::ScalarType dtype = static_cast<torch::ScalarType>(Data[offset] % 12); // 12 common dtypes
            try {
                torch::Tensor typed_input = input.to(dtype);
                torch::Tensor typed_result = torch::flatten(typed_input);
            } catch (...) {
                // Some dtype conversions might fail, that's okay
            }
            offset++;
        }

        // Test with zero-dimensional tensor
        torch::Tensor zero_dim = torch::tensor(42.0);
        torch::Tensor zero_dim_result = torch::flatten(zero_dim);

        // Test with 1D tensor (should return original or view)
        torch::Tensor one_dim = torch::randn({10});
        torch::Tensor one_dim_result = torch::flatten(one_dim);

        // Test with empty tensor
        if (offset < Size && (Data[offset] % 4) == 0) {
            std::vector<int64_t> empty_dims = dims;
            empty_dims[0] = 0; // Make first dimension 0
            torch::Tensor empty_tensor = torch::empty(empty_dims);
            torch::Tensor empty_result = torch::flatten(empty_tensor);
            offset++;
        }

        // Verify results have expected properties
        if (result1.numel() != input.numel()) {
            throw std::runtime_error("Flatten changed number of elements");
        }

        // Test contiguity and memory layout
        if (offset < Size) {
            // Test with non-contiguous tensor
            if (input.dim() >= 2) {
                torch::Tensor non_contiguous = input.transpose(0, 1);
                torch::Tensor nc_result = torch::flatten(non_contiguous);
            }
            offset++;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}