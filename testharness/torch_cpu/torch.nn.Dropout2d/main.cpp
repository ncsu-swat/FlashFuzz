#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 2D for Dropout2d
        // Dropout2d expects input of shape (N, C) or (N, C, H, W)
        if (input.dim() < 2) {
            // For 0D or 1D tensors, reshape to 2D (1, numel) tensor
            int64_t numel = input.numel();
            if (numel == 0) numel = 1;
            input = input.reshape({1, numel});
        }
        
        // Extract probability parameter from the input data
        float p = 0.5; // Default dropout probability
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            p = std::max(0.0f, std::min(1.0f, p));
            
            // Handle NaN/Inf
            if (!std::isfinite(p)) {
                p = 0.5f;
            }
        }
        
        // Extract inplace parameter from the input data
        bool inplace = false;
        if (offset < Size) {
            inplace = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Set training mode based on input data
        bool training = true;
        if (offset < Size) {
            training = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Clone input for inplace operations to avoid modifying original
        torch::Tensor input_for_module = inplace ? input.clone() : input;
        
        // Create Dropout2d module
        torch::nn::Dropout2d dropout_module(torch::nn::Dropout2dOptions().p(p).inplace(inplace));
        
        // Apply dropout in the appropriate mode
        torch::Tensor output;
        if (training) {
            dropout_module->train();
        } else {
            dropout_module->eval();
        }
        output = dropout_module->forward(input_for_module);
        
        // Access elements to ensure computation is performed
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test the functional interface as well with a fresh clone
        torch::Tensor input_for_functional = input.clone();
        torch::Tensor functional_output = torch::nn::functional::dropout2d(
            input_for_functional, 
            torch::nn::functional::Dropout2dFuncOptions().p(p).training(training)
        );
        
        if (functional_output.numel() > 0) {
            volatile float sum = functional_output.sum().item<float>();
            (void)sum;
        }
        
        // Test with different tensor dimensions (3D and 4D)
        if (input.dim() == 2 && input.numel() >= 4) {
            // Test 4D input (N, C, H, W)
            torch::Tensor input_4d = input.reshape({1, 1, -1, 1});
            torch::nn::Dropout2d dropout_4d(torch::nn::Dropout2dOptions().p(p));
            dropout_4d->train(training);
            torch::Tensor output_4d = dropout_4d->forward(input_4d);
            if (output_4d.numel() > 0) {
                volatile float sum = output_4d.sum().item<float>();
                (void)sum;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}