#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor creation and parameters
        if (Size < 16) {
            return 0;
        }

        // Extract tensor dimensions (1-6 dimensions)
        uint8_t num_dims = (Data[offset++] % 6) + 1;
        std::vector<int64_t> dims;
        
        for (int i = 0; i < num_dims && offset < Size; i++) {
            // Keep dimensions reasonable to avoid memory issues
            int64_t dim = (Data[offset++] % 10) + 1;
            dims.push_back(dim);
        }

        if (offset >= Size) return 0;

        // Extract start_dim and end_dim parameters
        int64_t start_dim = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));
        int64_t end_dim = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));

        // Create tensor with various data types
        torch::Tensor input_tensor;
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 6;
            switch (dtype_choice) {
                case 0:
                    input_tensor = torch::randn(dims, torch::kFloat32);
                    break;
                case 1:
                    input_tensor = torch::randn(dims, torch::kFloat64);
                    break;
                case 2:
                    input_tensor = torch::randint(-100, 100, dims, torch::kInt32);
                    break;
                case 3:
                    input_tensor = torch::randint(-100, 100, dims, torch::kInt64);
                    break;
                case 4:
                    input_tensor = torch::randint(0, 2, dims, torch::kBool);
                    break;
                default:
                    input_tensor = torch::randn(dims, torch::kFloat16);
                    break;
            }
        } else {
            input_tensor = torch::randn(dims);
        }

        // Test torch::flatten with different parameter combinations
        
        // Test 1: Basic flatten (no parameters)
        torch::Tensor result1 = torch::flatten(input_tensor);
        
        // Test 2: Flatten with start_dim only
        torch::Tensor result2 = torch::flatten(input_tensor, start_dim);
        
        // Test 3: Flatten with both start_dim and end_dim
        torch::Tensor result3 = torch::flatten(input_tensor, start_dim, end_dim);

        // Test edge cases with boundary values
        if (num_dims > 1) {
            // Test with valid boundary indices
            torch::Tensor result4 = torch::flatten(input_tensor, 0, num_dims - 1);
            torch::Tensor result5 = torch::flatten(input_tensor, -(int64_t)num_dims, -1);
            
            // Test with same start and end dim
            torch::Tensor result6 = torch::flatten(input_tensor, 0, 0);
            torch::Tensor result7 = torch::flatten(input_tensor, -1, -1);
        }

        // Test with different tensor layouts if possible
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            // Test with contiguous tensor
            torch::Tensor contiguous_tensor = input_tensor.contiguous();
            torch::Tensor result8 = torch::flatten(contiguous_tensor, start_dim, end_dim);
            
            // Test with transposed tensor (if 2D or higher)
            if (num_dims >= 2) {
                torch::Tensor transposed = input_tensor.transpose(0, 1);
                torch::Tensor result9 = torch::flatten(transposed, start_dim, end_dim);
            }
        }

        // Test with zero-sized tensors
        if (offset < Size && (Data[offset++] % 4 == 0)) {
            std::vector<int64_t> zero_dims = dims;
            if (!zero_dims.empty()) {
                zero_dims[0] = 0;  // Make first dimension zero
                torch::Tensor zero_tensor = torch::empty(zero_dims);
                torch::Tensor result10 = torch::flatten(zero_tensor);
            }
        }

        // Test with scalar tensor (0-dimensional)
        if (offset < Size && (Data[offset++] % 3 == 0)) {
            torch::Tensor scalar_tensor = torch::tensor(42.0);
            torch::Tensor result11 = torch::flatten(scalar_tensor);
        }

        // Verify results are valid tensors
        if (!result1.defined() || !result2.defined() || !result3.defined()) {
            return -1;
        }

        // Basic sanity checks
        if (result1.numel() != input_tensor.numel()) {
            return -1;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}