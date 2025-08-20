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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.detach operation
        torch::Tensor detached_tensor = input_tensor.detach();
        
        // Verify detached tensor properties
        if (detached_tensor.requires_grad()) {
            throw std::runtime_error("Detached tensor should not require gradients");
        }
        
        // Verify that detached tensor has the same data as the original
        if (!torch::allclose(input_tensor, detached_tensor)) {
            throw std::runtime_error("Detached tensor data differs from original tensor");
        }
        
        // Test detach on tensor that requires gradients
        if (input_tensor.is_floating_point()) {
            auto grad_tensor = input_tensor.clone().set_requires_grad(true);
            auto detached_grad_tensor = grad_tensor.detach();
            
            // Verify detached tensor doesn't require gradients
            if (detached_grad_tensor.requires_grad()) {
                throw std::runtime_error("Detached tensor from grad tensor should not require gradients");
            }
            
            // Verify data is the same
            if (!torch::allclose(grad_tensor, detached_grad_tensor)) {
                throw std::runtime_error("Detached tensor data differs from original tensor with gradients");
            }
        }
        
        // Test detach_() in-place operation
        if (offset + 1 < Size && Data[offset] % 2 == 0) {
            auto clone_tensor = input_tensor.clone();
            if (clone_tensor.is_floating_point()) {
                clone_tensor.set_requires_grad(true);
                clone_tensor.detach_();
                
                // Verify in-place detached tensor doesn't require gradients
                if (clone_tensor.requires_grad()) {
                    throw std::runtime_error("In-place detached tensor should not require gradients");
                }
                
                // Verify data is the same as original
                if (!torch::allclose(input_tensor, clone_tensor)) {
                    throw std::runtime_error("In-place detached tensor data differs from original tensor");
                }
            }
        }
        
        // Test detach on view
        if (offset + 1 < Size && input_tensor.numel() > 0 && input_tensor.dim() > 0) {
            auto view_tensor = input_tensor;
            if (view_tensor.is_floating_point()) {
                view_tensor.set_requires_grad(true);
                
                // Create a view
                auto view = view_tensor.slice(0, 0, view_tensor.size(0));
                
                // Detach the view
                auto detached_view = view.detach();
                
                // Verify detached view doesn't require gradients
                if (detached_view.requires_grad()) {
                    throw std::runtime_error("Detached view should not require gradients");
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