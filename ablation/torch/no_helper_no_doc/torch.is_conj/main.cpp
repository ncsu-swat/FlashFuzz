#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate a tensor with various properties to test is_conj
        auto tensor = generateTensor(Data, Size, offset);
        
        // Test basic is_conj functionality
        auto result = torch::is_conj(tensor);
        
        // Test with different tensor types and properties
        if (offset < Size) {
            // Test with complex tensors
            auto complex_tensor = tensor.to(torch::kComplexFloat);
            auto complex_result = torch::is_conj(complex_tensor);
            
            // Test with conjugated tensor
            auto conj_tensor = torch::conj(complex_tensor);
            auto conj_result = torch::is_conj(conj_tensor);
            
            // Test with double complex
            auto complex_double_tensor = tensor.to(torch::kComplexDouble);
            auto complex_double_result = torch::is_conj(complex_double_tensor);
            
            // Test with conjugated double complex tensor
            auto conj_double_tensor = torch::conj(complex_double_tensor);
            auto conj_double_result = torch::is_conj(conj_double_tensor);
        }
        
        // Test with different tensor shapes and sizes
        if (offset < Size) {
            auto reshaped_tensor = tensor.view({-1});
            auto reshaped_result = torch::is_conj(reshaped_tensor);
            
            // Test with scalar tensor
            auto scalar_tensor = torch::tensor(1.0, torch::kComplexFloat);
            auto scalar_result = torch::is_conj(scalar_tensor);
            
            // Test with conjugated scalar
            auto conj_scalar = torch::conj(scalar_tensor);
            auto conj_scalar_result = torch::is_conj(conj_scalar);
        }
        
        // Test with empty tensor
        auto empty_tensor = torch::empty({0}, torch::kComplexFloat);
        auto empty_result = torch::is_conj(empty_tensor);
        
        // Test with real tensors (should always return false)
        auto real_tensor = torch::randn({2, 3}, torch::kFloat);
        auto real_result = torch::is_conj(real_tensor);
        
        // Test with integer tensors (should always return false)
        auto int_tensor = torch::randint(0, 10, {2, 3}, torch::kInt);
        auto int_result = torch::is_conj(int_tensor);
        
        // Test edge cases with different devices if CUDA is available
        if (torch::cuda::is_available() && offset < Size) {
            auto cuda_tensor = tensor.to(torch::kCUDA).to(torch::kComplexFloat);
            auto cuda_result = torch::is_conj(cuda_tensor);
            
            auto cuda_conj_tensor = torch::conj(cuda_tensor);
            auto cuda_conj_result = torch::is_conj(cuda_conj_tensor);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}