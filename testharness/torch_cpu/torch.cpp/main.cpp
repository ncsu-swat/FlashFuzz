#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with max result

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
        
        // Skip if data is too small
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various torch operations
        
        // 1. Test basic operations
        torch::Tensor result1 = torch::abs(input);
        torch::Tensor result2 = torch::neg(input);
        torch::Tensor result3 = torch::square(input);
        
        // 2. Test reduction operations
        torch::Tensor sum_result = torch::sum(input);
        torch::Tensor mean_result = torch::mean(input.to(torch::kFloat));
        
        // 3. Test dimension-specific operations
        if (input.dim() > 0) {
            int64_t dim = 0;
            if (offset < Size) {
                dim = static_cast<int64_t>(Data[offset++]) % input.dim();
                try {
                    torch::Tensor dimSum = torch::sum(input, dim);
                    auto max_result = torch::max(input, dim);
                    torch::Tensor dimMax = std::get<0>(max_result);
                } catch (const std::exception&) {
                    // May fail for certain dtypes
                }
            }
        }
        
        // 4. Test type conversion
        torch::Tensor asFloat = input.to(torch::kFloat);
        try {
            torch::Tensor asInt = input.to(torch::kInt);
        } catch (const std::exception&) {
            // Complex to int conversion not supported
        }
        
        // 5. Test mathematical functions
        torch::Tensor float_input = input.to(torch::kFloat);
        torch::Tensor exp_result = torch::exp(float_input);
        torch::Tensor log_result = torch::log(torch::abs(float_input) + 1e-6f);
        
        // 6. Test tensor manipulation
        if (input.dim() > 0 && input.numel() > 0) {
            torch::Tensor transposed = input.transpose(0, input.dim() > 1 ? input.dim() - 1 : 0);
            torch::Tensor reshaped = input.reshape({-1});
            torch::Tensor cloned = input.clone();
        }
        
        // 7. Test binary operations
        torch::Tensor doubled = input + input;
        torch::Tensor squared = input * input;
        
        // 8. Test advanced operations
        if (input.dim() >= 2 && input.size(-1) > 0 && input.size(-2) > 0) {
            try {
                torch::Tensor float_in = input.to(torch::kFloat);
                torch::Tensor matmul_result = torch::matmul(float_in, float_in.transpose(-2, -1));
            } catch (const std::exception&) {
                // Matmul might fail for incompatible dimensions
            }
        }
        
        // 9. Test tensor creation functions
        torch::Tensor ones_like_result = torch::ones_like(input);
        torch::Tensor zeros_like_result = torch::zeros_like(input);
        
        // 10. Test more complex operations
        if (input.dim() > 0 && input.numel() > 0) {
            try {
                torch::Tensor softmax_result = torch::softmax(input.to(torch::kFloat), 0);
            } catch (const std::exception&) {
                // Softmax might fail for certain inputs
            }
            
            try {
                torch::Tensor norm_result = torch::norm(input.to(torch::kFloat));
            } catch (const std::exception&) {
                // Norm might fail for certain inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}