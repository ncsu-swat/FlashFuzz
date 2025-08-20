#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 4D for Dropout2d (batch_size, channels, height, width)
        // If not, reshape it to make it compatible
        if (input.dim() < 2) {
            // For 0D or 1D tensors, reshape to 2D (1, 1) tensor
            input = input.reshape({1, 1});
        }
        
        // Extract probability parameter from the input data
        float p = 0.5; // Default dropout probability
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            p = std::max(0.0f, std::min(1.0f, p));
        }
        
        // Extract inplace parameter from the input data
        bool inplace = false;
        if (offset < Size) {
            inplace = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Create Dropout2d module
        torch::nn::Dropout2d dropout_module(torch::nn::Dropout2dOptions().p(p).inplace(inplace));
        
        // Set training mode based on input data
        bool training = true;
        if (offset < Size) {
            training = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply dropout in the appropriate mode
        torch::Tensor output;
        if (training) {
            dropout_module->train();
            output = dropout_module->forward(input);
        } else {
            dropout_module->eval();
            output = dropout_module->forward(input);
        }
        
        // Test the functional interface as well
        torch::Tensor functional_output = torch::nn::functional::dropout2d(input, torch::nn::functional::Dropout2dFuncOptions().p(p).training(training));
        
        // Access elements to ensure computation is performed
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        if (functional_output.numel() > 0) {
            volatile float sum = functional_output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}