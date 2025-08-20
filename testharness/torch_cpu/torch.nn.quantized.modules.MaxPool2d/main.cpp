#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor has at least 3 dimensions (N, C, H, W) for MaxPool2d
        if (input_tensor.dim() < 3) {
            // Expand dimensions if needed
            while (input_tensor.dim() < 3) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Extract parameters for MaxPool2d from the remaining data
        uint8_t kernel_size = 2;
        uint8_t stride = 2;
        uint8_t padding = 0;
        uint8_t dilation = 1;
        bool ceil_mode = false;
        
        if (offset + 5 <= Size) {
            kernel_size = Data[offset++] % 5 + 1; // 1-5
            stride = Data[offset++] % 3 + 1;      // 1-3
            padding = Data[offset++] % 3;         // 0-2
            dilation = Data[offset++] % 2 + 1;    // 1-2
            ceil_mode = Data[offset++] % 2;       // 0-1
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + 8 <= Size) {
            // Extract scale (ensure it's positive)
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale);
            if (scale < 1e-6) scale = 1e-6; // Avoid extremely small scales
            if (scale > 1e6) scale = 1e6;   // Avoid extremely large scales
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            // Convert to float if not already
            if (input_tensor.scalar_type() != torch::kFloat) {
                input_tensor = input_tensor.to(torch::kFloat);
            }
            
            // Quantize the tensor
            quantized_input = torch::quantize_per_tensor(
                input_tensor, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({1, 3, 8, 8}, options);
            quantized_input = torch::quantize_per_tensor(
                simple_tensor, 1.0, 0, torch::kQUInt8);
        }
        
        // Apply quantized max_pool2d function directly
        torch::Tensor output;
        try {
            output = torch::max_pool2d(quantized_input, 
                                     {kernel_size, kernel_size},
                                     {stride, stride},
                                     {padding, padding},
                                     {dilation, dilation},
                                     ceil_mode);
        } catch (const std::exception& e) {
            // If operation fails, try with more reasonable parameters
            output = torch::max_pool2d(quantized_input, 
                                     {2, 2},
                                     {2, 2});
        }
        
        // Dequantize the output for verification
        torch::Tensor dequantized_output = output.dequantize();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}