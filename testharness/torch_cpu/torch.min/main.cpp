#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test different variants of torch::min
        
        // 1. Global min - returns a single value tensor
        try {
            torch::Tensor global_min = torch::min(input);
        } catch (...) {
            // May fail on empty tensors
        }
        
        // 2. Min along dimension with keepdim=false (default)
        if (offset < Size && input.dim() > 0) {
            int64_t dim = static_cast<int64_t>(Data[offset++]) % input.dim();
            try {
                std::tuple<torch::Tensor, torch::Tensor> result = torch::min(input, dim);
                torch::Tensor values = std::get<0>(result);
                torch::Tensor indices = std::get<1>(result);
            } catch (...) {
                // Ignore dimension-related errors
            }
        }
        
        // 3. Min along dimension with keepdim=true
        if (offset < Size && input.dim() > 0) {
            int64_t dim = static_cast<int64_t>(Data[offset++]) % input.dim();
            bool keepdim = offset < Size && (Data[offset++] % 2 == 0);
            try {
                std::tuple<torch::Tensor, torch::Tensor> result = torch::min(input, dim, keepdim);
                torch::Tensor values = std::get<0>(result);
                torch::Tensor indices = std::get<1>(result);
            } catch (...) {
                // Ignore dimension-related errors
            }
        }
        
        // 4. Element-wise min of two tensors using torch::minimum
        if (offset + 1 < Size) {
            torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                // torch::minimum is the correct API for element-wise min
                torch::Tensor elementwise_min = torch::minimum(input, other);
            } catch (...) {
                // Ignore shape mismatch errors
            }
        }
        
        // 5. Test with empty tensor
        if (offset < Size) {
            try {
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_tensor = torch::empty(empty_shape, input.options());
                torch::Tensor empty_min = torch::min(empty_tensor);
            } catch (...) {
                // Ignore errors with empty tensors (expected to fail)
            }
        }
        
        // 6. Test with scalar tensor
        if (offset < Size) {
            try {
                torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset++]));
                torch::Tensor scalar_min = torch::min(scalar_tensor);
            } catch (...) {
                // Ignore errors with scalar tensors
            }
        }
        
        // 7. Test with negative dimension
        if (offset < Size && input.dim() > 0) {
            try {
                int64_t neg_dim = -1 * (static_cast<int64_t>(Data[offset++]) % input.dim() + 1);
                std::tuple<torch::Tensor, torch::Tensor> result = torch::min(input, neg_dim);
            } catch (...) {
                // Ignore errors with negative dimensions
            }
        }
        
        // 8. Test with out parameter variants
        if (offset < Size && input.dim() > 0) {
            int64_t dim = static_cast<int64_t>(Data[offset++]) % input.dim();
            try {
                torch::Tensor values = torch::empty({}, input.options());
                torch::Tensor indices = torch::empty({}, torch::kLong);
                torch::min_out(values, indices, input, dim);
            } catch (...) {
                // Ignore errors with out parameters
            }
        }
        
        // 9. Test minimum with scalar
        if (offset < Size) {
            try {
                double scalar_val = static_cast<double>(Data[offset++]) / 128.0;
                torch::Tensor scalar_other = torch::full_like(input, scalar_val);
                torch::Tensor result = torch::minimum(input, scalar_other);
            } catch (...) {
                // Ignore errors
            }
        }
        
        // 10. Test with different dtypes
        if (offset < Size) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::Tensor float_min = torch::min(float_input);
                torch::Tensor double_min = torch::min(double_input);
            } catch (...) {
                // Ignore dtype conversion errors
            }
        }
        
        // 11. Test amin (alternative API for min along dimension without indices)
        if (offset < Size && input.dim() > 0) {
            int64_t dim = static_cast<int64_t>(Data[offset++]) % input.dim();
            try {
                torch::Tensor amin_result = torch::amin(input, {dim});
            } catch (...) {
                // Ignore errors
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