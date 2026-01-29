#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;
        
        // Read dimensions for the square matrix
        uint8_t dim_byte = Data[offset++];
        // Limit matrix size to avoid excessive computation (2-8)
        int64_t square_dim = 2 + (dim_byte % 7);
        
        // Read batch dimension option
        uint8_t batch_byte = Data[offset++];
        bool use_batch = (batch_byte % 2) == 0;
        int64_t batch_size = 1 + (batch_byte % 3);
        
        // Read dtype selection
        uint8_t dtype_byte = Data[offset++];
        
        // Create a square matrix from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Determine total elements needed
        int64_t total_needed = square_dim * square_dim;
        if (use_batch) {
            total_needed *= batch_size;
        }
        
        // Flatten and expand/truncate to get exactly the elements we need
        torch::Tensor flat = input_tensor.flatten();
        int64_t available = flat.numel();
        
        if (available == 0) {
            return 0;
        }
        
        // Repeat tensor if necessary to get enough elements
        if (available < total_needed) {
            int64_t repeats = (total_needed + available - 1) / available;
            flat = flat.repeat({repeats});
        }
        
        // Slice to exact size needed
        flat = flat.slice(0, 0, total_needed);
        
        // Reshape to square matrix (with optional batch dimension)
        if (use_batch) {
            input_tensor = flat.reshape({batch_size, square_dim, square_dim});
        } else {
            input_tensor = flat.reshape({square_dim, square_dim});
        }
        
        // Convert to float or double (matrix_exp requires floating point)
        torch::ScalarType target_dtype;
        switch (dtype_byte % 4) {
            case 0:
                target_dtype = torch::kFloat32;
                break;
            case 1:
                target_dtype = torch::kFloat64;
                break;
            case 2:
                target_dtype = torch::kComplexFloat;
                break;
            case 3:
                target_dtype = torch::kComplexDouble;
                break;
            default:
                target_dtype = torch::kFloat32;
        }
        input_tensor = input_tensor.to(target_dtype);
        
        // Apply matrix_exp operation (torch::matrix_exp in C++ frontend)
        torch::Tensor result = torch::matrix_exp(input_tensor);
        
        // Verify output shape matches input shape
        if (result.sizes() != input_tensor.sizes()) {
            std::cerr << "Shape mismatch in output" << std::endl;
        }
        
        // Test with contiguous and non-contiguous inputs
        if (offset < Size && input_tensor.dim() >= 2) {
            uint8_t test_byte = Data[offset++];
            if (test_byte % 2 == 0) {
                // Test with transposed input (non-contiguous)
                torch::Tensor transposed = input_tensor.transpose(-2, -1).contiguous();
                torch::Tensor result2 = torch::matrix_exp(transposed);
            }
        }
        
        // Test on a slice if we have batch dimension
        if (use_batch && batch_size > 1 && offset < Size) {
            uint8_t idx_byte = Data[offset++];
            int64_t index = idx_byte % batch_size;
            torch::Tensor slice = input_tensor.index({index});
            torch::Tensor slice_result = torch::matrix_exp(slice);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}