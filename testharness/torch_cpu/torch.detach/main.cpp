#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.detach operation
        torch::Tensor detached_tensor = input_tensor.detach();
        
        // Verify detached tensor does not require gradients
        assert(!detached_tensor.requires_grad());
        
        // Verify that detached tensor shares data with the original
        assert(torch::equal(input_tensor, detached_tensor));
        
        // Test detach on tensor that requires gradients (only floating point can have gradients)
        if (input_tensor.is_floating_point()) {
            auto grad_tensor = input_tensor.clone().set_requires_grad(true);
            auto detached_grad_tensor = grad_tensor.detach();
            
            // Verify detached tensor doesn't require gradients
            assert(!detached_grad_tensor.requires_grad());
            
            // Verify data is the same
            assert(torch::allclose(grad_tensor, detached_grad_tensor));
        }
        
        // Test detach_() in-place operation
        if (Size > offset && Data[offset % Size] % 2 == 0) {
            if (input_tensor.is_floating_point()) {
                auto clone_tensor = input_tensor.clone().set_requires_grad(true);
                clone_tensor.detach_();
                
                // Verify in-place detached tensor doesn't require gradients
                assert(!clone_tensor.requires_grad());
                
                // Verify data is the same as original
                assert(torch::allclose(input_tensor, clone_tensor));
            }
        }
        
        // Test detach on a view
        if (Size > offset && input_tensor.numel() > 0 && input_tensor.dim() > 0) {
            if (input_tensor.is_floating_point()) {
                auto base_tensor = input_tensor.clone().set_requires_grad(true);
                
                // Create a view via slice
                auto view = base_tensor.slice(0, 0, base_tensor.size(0));
                
                // Detach the view
                auto detached_view = view.detach();
                
                // Verify detached view doesn't require gradients
                assert(!detached_view.requires_grad());
            }
        }
        
        // Test detach on contiguous tensor
        if (Size > offset + 1 && Data[(offset + 1) % Size] % 3 == 0) {
            auto contiguous_tensor = input_tensor.contiguous();
            auto detached_contiguous = contiguous_tensor.detach();
            assert(torch::equal(contiguous_tensor, detached_contiguous));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}