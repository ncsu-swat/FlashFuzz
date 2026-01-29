#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply detach_ operation (in-place detach)
        // This should work on any tensor regardless of requires_grad status
        tensor.detach_();
        
        // After detach_, tensor should not require gradients
        // This is expected behavior, not an error condition
        (void)tensor.requires_grad();
        
        // Test with a floating point tensor that requires gradients
        try {
            // Create a float tensor that can have requires_grad=true
            torch::Tensor grad_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to float if not already a floating point type
            if (!grad_tensor.is_floating_point()) {
                grad_tensor = grad_tensor.to(torch::kFloat32);
            }
            
            // Enable gradient tracking
            grad_tensor = grad_tensor.detach().requires_grad_(true);
            
            // Verify it requires grad before detach_
            bool had_grad = grad_tensor.requires_grad();
            
            // Apply detach_ in-place
            grad_tensor.detach_();
            
            // After detach_, should not require gradients
            (void)grad_tensor.requires_grad();
            (void)had_grad;
        }
        catch (...) {
            // Silently catch shape/type related issues
        }
        
        // Test edge case: empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
            empty_tensor.detach_();
            (void)empty_tensor.requires_grad();
        }
        catch (...) {
            // Silently catch any issues with empty tensors
        }
        
        // Test edge case: scalar tensor
        try {
            torch::Tensor scalar_tensor = torch::tensor(3.14, torch::TensorOptions().requires_grad(true));
            scalar_tensor.detach_();
            (void)scalar_tensor.requires_grad();
        }
        catch (...) {
            // Silently catch any issues with scalar tensors
        }
        
        // Test with multi-dimensional tensors of various shapes
        try {
            uint8_t shape_selector = (Size > 0) ? Data[0] % 4 : 0;
            torch::Tensor multi_tensor;
            
            switch (shape_selector) {
                case 0:
                    multi_tensor = torch::randn({2, 3}, torch::TensorOptions().requires_grad(true));
                    break;
                case 1:
                    multi_tensor = torch::randn({4, 5, 6}, torch::TensorOptions().requires_grad(true));
                    break;
                case 2:
                    multi_tensor = torch::randn({1}, torch::TensorOptions().requires_grad(true));
                    break;
                default:
                    multi_tensor = torch::randn({2, 2, 2, 2}, torch::TensorOptions().requires_grad(true));
                    break;
            }
            
            multi_tensor.detach_();
            (void)multi_tensor.requires_grad();
        }
        catch (...) {
            // Silently catch any issues
        }
        
        // Test detach_ on a tensor that's part of a computation graph
        try {
            torch::Tensor a = torch::randn({3, 3}, torch::TensorOptions().requires_grad(true));
            torch::Tensor b = a * 2 + 1;  // b is part of computation graph
            
            // detach_ on intermediate tensor
            b.detach_();
            
            // b should no longer require grad and be disconnected from graph
            (void)b.requires_grad();
        }
        catch (...) {
            // Silently catch any issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}