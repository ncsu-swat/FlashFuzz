#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply detach operation (detach_copy doesn't exist, use detach().clone())
        torch::Tensor detached = input_tensor.detach().clone();
        
        // Verify that the detached tensor has the same data but doesn't share storage
        if (detached.sizes() != input_tensor.sizes() || 
            detached.dtype() != input_tensor.dtype()) {
            throw std::runtime_error("Detached tensor has different shape or dtype");
        }
        
        // Check that detached tensor doesn't require gradients
        if (detached.requires_grad()) {
            throw std::runtime_error("Detached tensor should not require gradients");
        }
        
        // Check that detached tensor has the same values
        if (!torch::allclose(detached, input_tensor)) {
            throw std::runtime_error("Detached tensor has different values");
        }
        
        // Test with gradients enabled on the input
        if (input_tensor.scalar_type() == torch::kFloat || 
            input_tensor.scalar_type() == torch::kDouble ||
            input_tensor.scalar_type() == torch::kHalf) {
            torch::Tensor grad_input = input_tensor.clone().set_requires_grad(true);
            torch::Tensor grad_detached = grad_input.detach().clone();
            
            // Verify detached tensor doesn't require gradients
            if (grad_detached.requires_grad()) {
                throw std::runtime_error("Detached tensor from grad-enabled input should not require gradients");
            }
            
            // Verify original tensor still requires gradients
            if (!grad_input.requires_grad()) {
                throw std::runtime_error("Original tensor should still require gradients");
            }
            
            // Modify detached tensor and verify it doesn't affect original
            if (grad_detached.numel() > 0) {
                grad_detached.fill_(0);
                if (torch::allclose(grad_input, grad_detached)) {
                    throw std::runtime_error("Modifying detached tensor affected original tensor");
                }
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