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
        if (shape.empty()) {
            return 0; // Invalid shape, discard input
        }

        auto dtype = parse_dtype(Data, Size, offset);
        if (dtype == torch::kNumOptions) {
            return 0; // Invalid dtype, discard input
        }

        // Create tensor with parsed shape and dtype
        torch::Tensor input_tensor = create_tensor(Data, Size, offset, shape, dtype);
        if (!input_tensor.defined()) {
            return 0; // Failed to create tensor, discard input
        }

        // Test basic count_nonzero without dimension specification
        torch::Tensor result1 = torch::count_nonzero(input_tensor);

        // Parse optional dimension parameter
        if (offset < Size) {
            int64_t dim_choice = parse_int(Data, Size, offset) % 3;
            
            if (dim_choice == 0 && input_tensor.dim() > 0) {
                // Test with single dimension
                int64_t dim = parse_int(Data, Size, offset) % input_tensor.dim();
                torch::Tensor result2 = torch::count_nonzero(input_tensor, dim);
            }
            else if (dim_choice == 1 && input_tensor.dim() > 1) {
                // Test with multiple dimensions
                std::vector<int64_t> dims;
                int num_dims = std::min(static_cast<int>(input_tensor.dim()), 
                                      static_cast<int>(parse_int(Data, Size, offset) % 4 + 1));
                
                for (int i = 0; i < num_dims && offset < Size; ++i) {
                    int64_t dim = parse_int(Data, Size, offset) % input_tensor.dim();
                    // Avoid duplicate dimensions
                    if (std::find(dims.begin(), dims.end(), dim) == dims.end()) {
                        dims.push_back(dim);
                    }
                }
                
                if (!dims.empty()) {
                    torch::Tensor result3 = torch::count_nonzero(input_tensor, dims);
                }
            }
        }

        // Test edge cases with special tensor types
        if (input_tensor.numel() > 0) {
            // Test with zero tensor
            torch::Tensor zero_tensor = torch::zeros_like(input_tensor);
            torch::Tensor zero_result = torch::count_nonzero(zero_tensor);

            // Test with ones tensor
            torch::Tensor ones_tensor = torch::ones_like(input_tensor);
            torch::Tensor ones_result = torch::count_nonzero(ones_tensor);

            // Test with tensor containing both zeros and non-zeros
            if (input_tensor.numel() >= 2) {
                torch::Tensor mixed_tensor = input_tensor.clone();
                mixed_tensor.flatten()[0] = 0;  // Set first element to zero
                if (mixed_tensor.numel() > 1) {
                    mixed_tensor.flatten()[1] = 1;  // Set second element to non-zero
                }
                torch::Tensor mixed_result = torch::count_nonzero(mixed_tensor);
            }
        }

        // Test with different tensor layouts if applicable
        if (input_tensor.dim() >= 2 && input_tensor.is_contiguous()) {
            try {
                torch::Tensor transposed = input_tensor.transpose(0, 1);
                torch::Tensor transposed_result = torch::count_nonzero(transposed);
            } catch (...) {
                // Ignore transpose errors for incompatible shapes
            }
        }

        // Test with negative dimensions if tensor has dimensions
        if (input_tensor.dim() > 0 && offset < Size) {
            int64_t neg_dim = -(parse_int(Data, Size, offset) % input_tensor.dim() + 1);
            torch::Tensor neg_result = torch::count_nonzero(input_tensor, neg_dim);
        }

        // Test with empty dimensions vector
        std::vector<int64_t> empty_dims;
        torch::Tensor empty_dims_result = torch::count_nonzero(input_tensor, empty_dims);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}