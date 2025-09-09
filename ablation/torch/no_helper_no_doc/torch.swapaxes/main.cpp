#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor creation and axis parameters
        if (Size < 16) {
            return 0;
        }

        // Generate tensor dimensions (1-6 dimensions)
        int num_dims = (Data[offset] % 6) + 1;
        offset++;

        std::vector<int64_t> dims;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            int64_t dim_size = (Data[offset] % 10) + 1; // 1-10 size per dimension
            dims.push_back(dim_size);
            offset++;
        }

        if (dims.empty()) {
            dims.push_back(1); // Fallback to 1D tensor
        }

        // Create input tensor with random data
        torch::Tensor input = torch::randn(dims);

        // Get axis0 and axis1 for swapping
        if (offset + 1 >= Size) {
            return 0;
        }

        int64_t axis0 = static_cast<int64_t>(static_cast<int8_t>(Data[offset])) % static_cast<int64_t>(dims.size());
        offset++;
        int64_t axis1 = static_cast<int64_t>(static_cast<int8_t>(Data[offset])) % static_cast<int64_t>(dims.size());
        offset++;

        // Test torch::swapaxes with the generated parameters
        torch::Tensor result = torch::swapaxes(input, axis0, axis1);

        // Verify result properties
        if (result.numel() != input.numel()) {
            std::cerr << "swapaxes changed tensor size unexpectedly" << std::endl;
        }

        // Test with negative axis values
        int64_t neg_axis0 = axis0 - static_cast<int64_t>(dims.size());
        int64_t neg_axis1 = axis1 - static_cast<int64_t>(dims.size());
        torch::Tensor result_neg = torch::swapaxes(input, neg_axis0, neg_axis1);

        // Test edge cases
        if (dims.size() >= 2) {
            // Test swapping same axis
            torch::Tensor same_axis = torch::swapaxes(input, 0, 0);
            
            // Test swapping first and last dimensions
            torch::Tensor first_last = torch::swapaxes(input, 0, static_cast<int64_t>(dims.size()) - 1);
        }

        // Test with different tensor types if we have enough data
        if (offset < Size) {
            torch::ScalarType dtype = static_cast<torch::ScalarType>((Data[offset] % 12) + 1); // Various dtypes
            try {
                torch::Tensor typed_input = input.to(dtype);
                torch::Tensor typed_result = torch::swapaxes(typed_input, axis0, axis1);
            } catch (const std::exception&) {
                // Some dtype conversions might fail, that's okay
            }
            offset++;
        }

        // Test with contiguous and non-contiguous tensors
        if (input.dim() >= 2) {
            torch::Tensor transposed = input.transpose(0, 1);
            torch::Tensor non_contiguous_result = torch::swapaxes(transposed, axis0, axis1);
        }

        // Test boundary conditions
        if (dims.size() > 1) {
            // Test with maximum valid axis values
            int64_t max_axis = static_cast<int64_t>(dims.size()) - 1;
            torch::Tensor max_result = torch::swapaxes(input, 0, max_axis);
            
            // Test with minimum valid axis values (negative)
            torch::Tensor min_result = torch::swapaxes(input, -1, -static_cast<int64_t>(dims.size()));
        }

        // Test with empty tensor if dimensions allow
        if (offset < Size && (Data[offset] % 10) == 0) {
            std::vector<int64_t> empty_dims = dims;
            if (!empty_dims.empty()) {
                empty_dims[0] = 0; // Make first dimension empty
                torch::Tensor empty_tensor = torch::zeros(empty_dims);
                torch::Tensor empty_result = torch::swapaxes(empty_tensor, axis0, axis1);
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