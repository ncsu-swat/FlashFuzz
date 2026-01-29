#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // corrcoef expects 1D or 2D tensor
        // For 2D: rows are variables, columns are observations
        // For 1D: treated as single variable with multiple observations
        
        // Reshape to valid dimensions if needed
        torch::Tensor valid_tensor;
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }
        
        if (input_tensor.dim() == 1) {
            valid_tensor = input_tensor;
        } else if (input_tensor.dim() == 2) {
            valid_tensor = input_tensor;
        } else {
            // Flatten high-dimensional tensor to 2D
            int64_t total = input_tensor.numel();
            int64_t rows = input_tensor.size(0);
            int64_t cols = total / rows;
            if (cols == 0) {
                cols = 1;
                rows = total;
            }
            valid_tensor = input_tensor.reshape({rows, cols});
        }
        
        // Ensure floating point type for corrcoef
        if (!valid_tensor.is_floating_point() && !valid_tensor.is_complex()) {
            valid_tensor = valid_tensor.to(torch::kFloat32);
        }
        
        // Apply torch.corrcoef operation
        try {
            torch::Tensor result = torch::corrcoef(valid_tensor);
        } catch (const std::exception&) {
            // May throw for certain edge cases, continue
        }
        
        // Try with float64 for better numerical precision
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                torch::Tensor double_tensor = valid_tensor.to(torch::kFloat64);
                torch::Tensor result_double = torch::corrcoef(double_tensor);
            } catch (const std::exception&) {
                // Continue on error
            }
        }
        
        // Try with 1D tensor explicitly
        if (offset + 1 < Size) {
            int64_t len = std::max(int64_t(2), int64_t(Data[offset] % 32 + 2));
            torch::Tensor tensor_1d = torch::randn({len});
            try {
                torch::Tensor result_1d = torch::corrcoef(tensor_1d);
            } catch (const std::exception&) {
                // Continue on error
            }
            offset++;
        }
        
        // Try with tensor containing NaN/Inf values
        if (offset + 1 < Size) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor special_tensor;
            
            uint8_t choice = Data[offset] % 4;
            offset++;
            
            if (choice == 0) {
                // Create tensor with NaN
                special_tensor = torch::randn({3, 5}, options);
                special_tensor.index_put_({0, 0}, std::numeric_limits<float>::quiet_NaN());
            } else if (choice == 1) {
                // Create tensor with Inf
                special_tensor = torch::randn({3, 5}, options);
                special_tensor.index_put_({0, 0}, std::numeric_limits<float>::infinity());
            } else if (choice == 2) {
                // Create tensor with -Inf
                special_tensor = torch::randn({3, 5}, options);
                special_tensor.index_put_({0, 0}, -std::numeric_limits<float>::infinity());
            } else {
                // Create tensor with mixed special values
                special_tensor = torch::randn({4, 6}, options);
                special_tensor.index_put_({0, 0}, std::numeric_limits<float>::quiet_NaN());
                special_tensor.index_put_({1, 1}, std::numeric_limits<float>::infinity());
            }
            
            try {
                torch::Tensor result_special = torch::corrcoef(special_tensor);
            } catch (const std::exception&) {
                // May throw for invalid values, continue
            }
        }
        
        // Try with complex tensor
        if (offset < Size && Data[offset] % 3 == 0) {
            try {
                torch::Tensor complex_tensor = torch::randn({2, 4}, torch::kComplexFloat);
                torch::Tensor result_complex = torch::corrcoef(complex_tensor);
            } catch (const std::exception&) {
                // Complex may not be supported, continue
            }
        }
        
        // Try with single row/column
        if (offset + 1 < Size) {
            int64_t cols = std::max(int64_t(2), int64_t(Data[offset] % 16 + 2));
            try {
                torch::Tensor single_row = torch::randn({1, cols});
                torch::Tensor result_single = torch::corrcoef(single_row);
            } catch (const std::exception&) {
                // Continue on error
            }
        }
        
        // Try with larger matrix
        if (offset + 2 < Size) {
            int64_t rows = std::max(int64_t(2), int64_t(Data[offset] % 10 + 2));
            int64_t cols = std::max(int64_t(2), int64_t(Data[offset + 1] % 20 + 2));
            try {
                torch::Tensor large_tensor = torch::randn({rows, cols});
                torch::Tensor result_large = torch::corrcoef(large_tensor);
            } catch (const std::exception&) {
                // Continue on error
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}