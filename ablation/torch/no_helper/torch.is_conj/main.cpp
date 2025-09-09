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
        
        // Test is_conj on the original tensor
        bool is_conj_result = torch::is_conj(tensor);
        
        // Test with conjugated tensor if it's a complex tensor
        if (tensor.is_complex()) {
            auto conj_tensor = torch::conj(tensor);
            bool conj_is_conj_result = torch::is_conj(conj_tensor);
            
            // Test with physical conjugate (resolve_conj)
            auto physical_conj = torch::resolve_conj(conj_tensor);
            bool physical_is_conj_result = torch::is_conj(physical_conj);
            
            // Test chaining conjugations
            auto double_conj = torch::conj(conj_tensor);
            bool double_conj_result = torch::is_conj(double_conj);
        }
        
        // Test with different tensor types and shapes
        if (tensor.numel() > 0) {
            // Test with reshaped tensor
            auto reshaped = tensor.reshape({-1});
            bool reshaped_result = torch::is_conj(reshaped);
            
            // Test with sliced tensor
            auto sliced = tensor.slice(0, 0, std::min(tensor.size(0), static_cast<int64_t>(2)));
            bool sliced_result = torch::is_conj(sliced);
        }
        
        // Test with cloned tensor
        auto cloned = tensor.clone();
        bool cloned_result = torch::is_conj(cloned);
        
        // Test with detached tensor
        auto detached = tensor.detach();
        bool detached_result = torch::is_conj(detached);
        
        // Test edge cases with empty tensors
        auto empty_tensor = torch::empty({0}, tensor.options());
        bool empty_result = torch::is_conj(empty_tensor);
        
        // Test with scalar tensor
        auto scalar_tensor = torch::tensor(1.0, tensor.options());
        bool scalar_result = torch::is_conj(scalar_tensor);
        
        // If complex, test conjugate of scalar
        if (scalar_tensor.is_complex()) {
            auto conj_scalar = torch::conj(scalar_tensor);
            bool conj_scalar_result = torch::is_conj(conj_scalar);
        }
        
        // Test with different memory formats if applicable
        if (tensor.dim() >= 2 && tensor.numel() > 0) {
            try {
                auto contiguous_tensor = tensor.contiguous();
                bool contiguous_result = torch::is_conj(contiguous_tensor);
                
                if (contiguous_tensor.is_complex()) {
                    auto conj_contiguous = torch::conj(contiguous_tensor);
                    bool conj_contiguous_result = torch::is_conj(conj_contiguous);
                }
            } catch (...) {
                // Ignore memory format errors
            }
        }
        
        // Test with view operations
        if (tensor.numel() > 1) {
            try {
                auto view_tensor = tensor.view({-1});
                bool view_result = torch::is_conj(view_tensor);
                
                if (view_tensor.is_complex()) {
                    auto conj_view = torch::conj(view_tensor);
                    bool conj_view_result = torch::is_conj(conj_view);
                }
            } catch (...) {
                // Ignore view errors for incompatible shapes
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