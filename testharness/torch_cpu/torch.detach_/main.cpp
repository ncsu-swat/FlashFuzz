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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply detach_ operation (in-place detach)
        tensor.detach_();
        
        // Verify that detach_ worked correctly
        // 1. The tensor should not require gradients
        if (tensor.requires_grad()) {
            throw std::runtime_error("detach_ failed: tensor still requires gradients");
        }
        
        // 2. The tensor should share the same storage as the original
        if (!tensor.is_same_size(original)) {
            throw std::runtime_error("detach_ changed tensor size");
        }
        
        // 3. The tensor data should be identical to the original
        if (!torch::all(tensor.eq(original)).item<bool>()) {
            throw std::runtime_error("detach_ changed tensor data");
        }
        
        // Try with requires_grad=true
        if (offset + 1 < Size) {
            // Create a new tensor that requires gradients
            torch::Tensor grad_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            grad_tensor.set_requires_grad(true);
            
            // Store original data
            torch::Tensor grad_original = grad_tensor.clone();
            
            // Apply detach_
            grad_tensor.detach_();
            
            // Verify detach_ worked correctly
            if (grad_tensor.requires_grad()) {
                throw std::runtime_error("detach_ failed on requires_grad=true tensor");
            }
            
            if (!torch::all(grad_tensor.eq(grad_original)).item<bool>()) {
                throw std::runtime_error("detach_ changed tensor data for requires_grad=true tensor");
            }
        }
        
        // Test edge case: empty tensor
        if (offset + 1 < Size) {
            torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().requires_grad(true));
            empty_tensor.detach_();
            if (empty_tensor.requires_grad()) {
                throw std::runtime_error("detach_ failed on empty tensor");
            }
        }
        
        // Test edge case: scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(3.14, torch::TensorOptions().requires_grad(true));
            scalar_tensor.detach_();
            if (scalar_tensor.requires_grad()) {
                throw std::runtime_error("detach_ failed on scalar tensor");
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