#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to test with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch.is_inference mode
        bool initial_mode = torch::autograd::is_inference_mode_enabled();
        
        // Test enabling inference mode
        torch::InferenceMode guard(true);
        bool enabled_mode = torch::autograd::is_inference_mode_enabled();
        
        // Test disabling inference mode
        torch::InferenceMode guard2(false);
        bool disabled_mode = torch::autograd::is_inference_mode_enabled();
        
        // Test with tensor operations in inference mode
        torch::InferenceMode inference_on(true);
        torch::Tensor result1 = tensor + 1;
        
        // Test with tensor operations outside inference mode
        torch::InferenceMode inference_off(false);
        torch::Tensor result2 = tensor + 2;
        
        // Test nested inference modes
        {
            torch::InferenceMode outer(true);
            bool outer_mode = torch::autograd::is_inference_mode_enabled();
            
            {
                torch::InferenceMode inner(false);
                bool inner_mode = torch::autograd::is_inference_mode_enabled();
            }
            
            bool after_inner_mode = torch::autograd::is_inference_mode_enabled();
        }
        
        // Test with tensor creation in inference mode
        torch::InferenceMode creation_guard(true);
        torch::Tensor new_tensor = torch::ones_like(tensor);
        
        // Test with tensor requires_grad in inference mode
        torch::InferenceMode grad_guard(true);
        torch::Tensor grad_tensor = tensor.clone();
        grad_tensor.set_requires_grad(true);
        
        // Test with autograd operations
        torch::InferenceMode autograd_guard(true);
        torch::Tensor autograd_tensor = tensor.clone();
        if (autograd_tensor.requires_grad()) {
            autograd_tensor = autograd_tensor * 2;
        }
        
        // Test with different tensor types
        if (offset + 1 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            torch::InferenceMode type_guard(true);
            torch::Tensor combined = tensor + another_tensor;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}