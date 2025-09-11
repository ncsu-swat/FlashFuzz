#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to test factory_kwargs
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract some parameters from the remaining data
        bool requires_grad = false;
        bool pin_memory = false;
        
        if (offset < Size) {
            requires_grad = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            pin_memory = Data[offset++] & 0x1;
        }
        
        // Create factory_kwargs options
        auto options = torch::TensorOptions()
            .dtype(tensor.dtype())
            .device(tensor.device())
            .requires_grad(requires_grad)
            .pinned_memory(pin_memory);
        
        // Test factory_kwargs by creating a new tensor with the same data but different options
        torch::Tensor result = torch::empty_like(tensor, options);
        
        // Copy data from original tensor to the new one
        result.copy_(tensor);
        
        // Verify the new tensor has the expected properties
        if (result.requires_grad() != requires_grad) {
            throw std::runtime_error("requires_grad mismatch");
        }
        
        if (result.is_pinned() != pin_memory) {
            throw std::runtime_error("pin_memory mismatch");
        }
        
        // Test other factory functions with the same options
        torch::Tensor zeros = torch::zeros_like(tensor, options);
        torch::Tensor ones = torch::ones_like(tensor, options);
        torch::Tensor rand = torch::rand_like(tensor, options);
        
        // Test with explicit factory_kwargs
        torch::nn::functional::BatchNormFuncOptions bn_options = 
            torch::nn::functional::BatchNormFuncOptions()
            .momentum(0.1)
            .eps(1e-5)
            .training(true);
        
        // Only apply batch norm if tensor has appropriate dimensions
        if (tensor.dim() >= 2 && tensor.size(1) > 0) {
            try {
                torch::Tensor running_mean = torch::zeros({tensor.size(1)});
                torch::Tensor running_var = torch::ones({tensor.size(1)});
                auto bn_result = torch::nn::functional::batch_norm(tensor, running_mean, running_var, bn_options);
            } catch (...) {
                // Batch norm might fail for various reasons, that's fine
            }
        }
        
        // Test with randn and factory_kwargs
        try {
            auto randn_tensor = torch::randn(tensor.sizes(), options);
        } catch (...) {
            // This might fail for some dtypes, that's expected
        }
        
        // Test with full and factory_kwargs
        try {
            auto full_tensor = torch::full(tensor.sizes(), 3.14, options);
        } catch (...) {
            // This might fail for some dtypes, that's expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
