#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if data is too small
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various torch.cpp operations
        
        // 1. Test basic operations
        torch::Tensor result1 = torch::abs(input);
        torch::Tensor result2 = torch::neg(input);
        torch::Tensor result3 = torch::square(input);
        
        // 2. Test reduction operations
        torch::Tensor sum = torch::sum(input);
        torch::Tensor mean = torch::mean(input);
        
        // 3. Test dimension-specific operations
        if (input.dim() > 0) {
            int64_t dim = 0;
            if (offset < Size) {
                dim = static_cast<int64_t>(Data[offset++]) % input.dim();
                torch::Tensor dimSum = torch::sum(input, dim);
                auto max_result = torch::max(input, dim);
                torch::Tensor dimMax = std::get<0>(max_result);
            }
        }
        
        // 4. Test type conversion
        torch::Tensor asFloat = input.to(torch::kFloat);
        torch::Tensor asInt = input.to(torch::kInt);
        
        // 5. Test mathematical functions
        torch::Tensor exp_result = torch::exp(input.to(torch::kFloat));
        torch::Tensor log_result = torch::log(torch::abs(input.to(torch::kFloat)) + 1e-6);
        
        // 6. Test tensor manipulation
        if (input.dim() > 0 && input.numel() > 0) {
            torch::Tensor transposed = input.transpose(0, input.dim() - 1);
            torch::Tensor reshaped = input.reshape({-1});
            torch::Tensor cloned = input.clone();
        }
        
        // 7. Test binary operations
        torch::Tensor doubled = input + input;
        torch::Tensor squared = input * input;
        
        // 8. Test advanced operations
        if (input.dim() >= 2) {
            try {
                torch::Tensor matmul_result = torch::matmul(input, input.transpose(-2, -1));
            } catch (const std::exception&) {
                // Matmul might fail for incompatible dimensions, which is expected
            }
        }
        
        // 9. Test tensor creation functions
        torch::Tensor ones_like = torch::ones_like(input);
        torch::Tensor zeros_like = torch::zeros_like(input);
        
        // 10. Test more complex operations
        if (input.dim() > 0 && input.numel() > 0) {
            try {
                torch::Tensor softmax = torch::softmax(input.to(torch::kFloat), 0);
            } catch (const std::exception&) {
                // Softmax might fail for certain inputs, which is expected
            }
            
            try {
                torch::Tensor norm = torch::norm(input.to(torch::kFloat));
            } catch (const std::exception&) {
                // Norm might fail for certain inputs, which is expected
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}