#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract tensor properties
        auto shape = extract_tensor_shape(Data, Size, offset, 1, 6); // 1-6 dimensions
        auto dtype = extract_dtype(Data, Size, offset);
        
        // Create input tensor with random values
        torch::Tensor input = create_tensor(shape, dtype);
        
        // Fill tensor with mix of zero and non-zero values to make the test meaningful
        if (input.numel() > 0) {
            auto flat_input = input.flatten();
            for (int64_t i = 0; i < flat_input.numel(); ++i) {
                uint8_t val_byte = extract_byte(Data, Size, offset);
                // Make roughly half the values zero, half non-zero
                if (val_byte % 2 == 0) {
                    flat_input[i] = 0;
                } else {
                    // Set to non-zero value based on dtype
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        flat_input[i] = (val_byte % 10) + 0.1; // Non-zero float
                    } else if (dtype == torch::kBool) {
                        flat_input[i] = true;
                    } else {
                        flat_input[i] = (val_byte % 10) + 1; // Non-zero integer
                    }
                }
            }
        }

        // Test 1: count_nonzero without dim (count all non-zeros)
        torch::Tensor result1 = torch::count_nonzero(input);
        
        // Verify result is a scalar
        if (result1.dim() != 0) {
            std::cerr << "count_nonzero without dim should return scalar" << std::endl;
        }

        // Test 2: count_nonzero with single dimension
        if (input.dim() > 0) {
            uint8_t dim_byte = extract_byte(Data, Size, offset);
            int64_t dim = dim_byte % input.dim();
            
            torch::Tensor result2 = torch::count_nonzero(input, dim);
            
            // Verify result shape
            auto expected_shape = input.sizes().vec();
            expected_shape.erase(expected_shape.begin() + dim);
            if (result2.sizes().vec() != expected_shape) {
                std::cerr << "count_nonzero with dim has unexpected shape" << std::endl;
            }
        }

        // Test 3: count_nonzero with multiple dimensions (if tensor has enough dims)
        if (input.dim() >= 2) {
            uint8_t dim1_byte = extract_byte(Data, Size, offset);
            uint8_t dim2_byte = extract_byte(Data, Size, offset);
            
            int64_t dim1 = dim1_byte % input.dim();
            int64_t dim2 = dim2_byte % input.dim();
            
            // Ensure different dimensions
            if (dim1 != dim2) {
                std::vector<int64_t> dims = {dim1, dim2};
                torch::Tensor result3 = torch::count_nonzero(input, dims);
                
                // Result should have reduced dimensions
                if (result3.dim() > input.dim() - 2) {
                    std::cerr << "count_nonzero with multiple dims has unexpected dimensionality" << std::endl;
                }
            }
        }

        // Test 4: Edge cases with negative dimensions
        if (input.dim() > 0) {
            uint8_t neg_dim_byte = extract_byte(Data, Size, offset);
            int64_t neg_dim = -1 - (neg_dim_byte % input.dim());
            
            torch::Tensor result4 = torch::count_nonzero(input, neg_dim);
            
            // Should work the same as positive dimension
            int64_t pos_dim = input.dim() + neg_dim;
            torch::Tensor result4_pos = torch::count_nonzero(input, pos_dim);
            
            if (!torch::equal(result4, result4_pos)) {
                std::cerr << "Negative dimension indexing inconsistent" << std::endl;
            }
        }

        // Test 5: All dimensions (should be equivalent to no dim specified)
        if (input.dim() > 0) {
            std::vector<int64_t> all_dims;
            for (int64_t i = 0; i < input.dim(); ++i) {
                all_dims.push_back(i);
            }
            
            torch::Tensor result5 = torch::count_nonzero(input, all_dims);
            
            // Should be scalar and equal to result1
            if (result5.dim() != 0 || !torch::equal(result5, result1)) {
                std::cerr << "count_nonzero with all dims should equal no dim case" << std::endl;
            }
        }

        // Test 6: Empty tensor
        if (input.numel() == 0) {
            torch::Tensor empty_result = torch::count_nonzero(input);
            if (empty_result.item<int64_t>() != 0) {
                std::cerr << "Empty tensor should have zero non-zero count" << std::endl;
            }
        }

        // Test 7: Tensor with all zeros
        torch::Tensor zeros_tensor = torch::zeros_like(input);
        torch::Tensor zeros_result = torch::count_nonzero(zeros_tensor);
        if (zeros_result.item<int64_t>() != 0) {
            std::cerr << "All-zeros tensor should have zero non-zero count" << std::endl;
        }

        // Test 8: Tensor with all non-zeros (if possible for the dtype)
        if (dtype != torch::kBool) { // Bool can only be 0 or 1
            torch::Tensor ones_tensor = torch::ones_like(input);
            torch::Tensor ones_result = torch::count_nonzero(ones_tensor);
            if (ones_result.item<int64_t>() != input.numel()) {
                std::cerr << "All-ones tensor should have count equal to numel" << std::endl;
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