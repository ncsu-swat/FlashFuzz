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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for skip_init
        bool skip_init = false;
        if (offset < Size) {
            skip_init = Data[offset++] & 0x1; // Use lowest bit to determine skip_init
        }
        
        // Create a simple module to test skip_init
        torch::nn::Linear model = torch::nn::Linear(
            torch::nn::LinearOptions(input_tensor.size(-1), 10).bias(true)
        );
        
        // Apply skip_init - torch::nn::utils::skip_init doesn't exist in PyTorch C++
        // Instead, we can manually set parameters to uninitialized state or skip initialization
        if (skip_init) {
            // Manually uninitialize parameters by setting them to empty tensors
            for (auto& param : model->parameters()) {
                param.detach_();
            }
        }
        
        // Test the model with the input tensor
        if (input_tensor.dim() > 0 && input_tensor.size(-1) > 0) {
            try {
                auto output = model->forward(input_tensor);
            } catch (const std::exception& e) {
                // Forward might fail for invalid inputs, but that's expected
            }
        }
        
        // Test other module types with skip_init
        torch::nn::Conv2d conv_model = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 16, 3).stride(1).padding(1)
        );
        
        if (skip_init) {
            // Manually uninitialize parameters
            for (auto& param : conv_model->parameters()) {
                param.detach_();
            }
        }
        
        // Create a sequential model and test skip_init
        auto seq_model = torch::nn::Sequential(
            torch::nn::Linear(10, 20),
            torch::nn::ReLU(),
            torch::nn::Linear(20, 5)
        );
        
        if (skip_init) {
            // Manually uninitialize parameters
            for (auto& param : seq_model->parameters()) {
                param.detach_();
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
