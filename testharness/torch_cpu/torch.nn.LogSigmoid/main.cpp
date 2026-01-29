#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create LogSigmoid module
        torch::nn::LogSigmoid logsigmoid;
        
        // Apply LogSigmoid to the input tensor using the module
        torch::Tensor output = logsigmoid(input);
        
        // Access the output to ensure computation happens
        volatile float first_elem = 0.0f;
        if (output.numel() > 0) {
            first_elem = output.flatten()[0].item<float>();
        }
        (void)first_elem;
        
        // Alternative way to apply LogSigmoid using functional API
        torch::Tensor output2 = torch::log_sigmoid(input);
        
        // Access second output
        if (output2.numel() > 0) {
            volatile float elem = output2.flatten()[0].item<float>();
            (void)elem;
        }
        
        // Test with different input types if we have more data
        if (offset + 1 < Size) {
            size_t new_offset = 0;
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, new_offset);
            
            // Apply LogSigmoid to the second input tensor
            torch::Tensor output3 = logsigmoid(input2);
            
            // Access the output
            if (output3.numel() > 0) {
                volatile float elem = output3.flatten()[0].item<float>();
                (void)elem;
            }
        }
        
        // Test with specific tensor configurations to improve coverage
        try {
            // Test with a float tensor
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor float_output = logsigmoid(float_input);
            (void)float_output;
        } catch (...) {
            // Silently catch dtype conversion issues
        }
        
        try {
            // Test with a double tensor
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor double_output = logsigmoid(double_input);
            (void)double_output;
        } catch (...) {
            // Silently catch dtype conversion issues
        }
        
        // Test with contiguous and non-contiguous tensors
        try {
            if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor output_t = logsigmoid(transposed);
                (void)output_t;
            }
        } catch (...) {
            // Silently catch shape-related issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}