#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight, bias, running_mean, and running_var tensors
        // These should be 1D tensors with size matching the number of channels (dim 1) of input
        int64_t num_channels = 1;
        if (input.dim() > 1) {
            num_channels = input.size(1);
        }
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + 8 <= Size) {
            // Extract scale from input data
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Make sure scale is positive and not too large
            scale = std::abs(scale);
            if (scale < 1e-6) scale = 1e-6;
            if (scale > 1e6) scale = 1e6;
        }
        
        if (offset + 8 <= Size) {
            // Extract zero_point from input data
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Limit zero_point to reasonable range
            zero_point = zero_point % 256;
        }
        
        // Create quantized tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            // If quantization fails, try with a default tensor
            q_input = torch::quantize_per_tensor(torch::ones({1, num_channels, 1, 1}), scale, zero_point, torch::kQUInt8);
        }
        
        // Create weight, bias, running_mean, and running_var
        torch::Tensor weight = torch::ones({num_channels});
        torch::Tensor bias = torch::zeros({num_channels});
        torch::Tensor running_mean = torch::zeros({num_channels});
        torch::Tensor running_var = torch::ones({num_channels});
        
        // Extract epsilon and momentum from input data if available
        double epsilon = 1e-5;
        double momentum = 0.1;
        
        if (offset + 8 <= Size) {
            std::memcpy(&epsilon, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure epsilon is positive
            epsilon = std::abs(epsilon);
            if (epsilon < 1e-10) epsilon = 1e-10;
            if (epsilon > 0.1) epsilon = 0.1;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = 1.0;
        }
        
        // Apply quantized_batch_norm
        torch::Tensor output;
        try {
            output = torch::quantized_batch_norm(
                q_input, 
                weight, 
                bias, 
                running_mean, 
                running_var, 
                epsilon, 
                scale, 
                zero_point
            );
        } catch (const std::exception& e) {
            // If the operation fails, just return
            return 0;
        }
        
        // Try to access output tensor properties to ensure it's valid
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try dequantizing the output
        try {
            torch::Tensor dequantized = output.dequantize();
        } catch (...) {
            // Ignore dequantization errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}