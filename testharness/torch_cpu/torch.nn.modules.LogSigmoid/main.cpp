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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create LogSigmoid module
        torch::nn::LogSigmoid logsigmoid;
        
        // Apply LogSigmoid to the input tensor
        torch::Tensor output = logsigmoid->forward(input);
        
        // Verify the output is valid
        if (output.numel() != input.numel()) {
            throw std::runtime_error("Output tensor has different number of elements than input tensor");
        }
        
        // Alternative implementation to verify correctness
        torch::Tensor expected_output = torch::log(torch::sigmoid(input));
        
        // Compare the results
        fuzzer_utils::compareTensors(output, expected_output, Data, Size);
        
        // Try backward pass if we have enough data and tensor requires grad
        if (offset < Size && Size - offset >= 1) {
            // Create a tensor that requires gradients
            auto input_with_grad = input.clone().detach().requires_grad_(true);
            
            // Forward pass
            auto output_with_grad = logsigmoid->forward(input_with_grad);
            
            // Create a gradient tensor
            auto grad_output = torch::ones_like(output_with_grad);
            
            // Backward pass
            output_with_grad.backward(grad_output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}