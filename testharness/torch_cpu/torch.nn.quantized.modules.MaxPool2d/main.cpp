#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        if (Size < 10) {
            return 0;
        }
        
        // Extract parameters for MaxPool2d first
        uint8_t kernel_size = (Data[offset++] % 4) + 1; // 1-4
        uint8_t stride = (Data[offset++] % 3) + 1;      // 1-3
        uint8_t padding = Data[offset++] % 2;           // 0-1
        uint8_t dilation = 1;                           // Keep dilation at 1 for simplicity
        bool ceil_mode = Data[offset++] % 2;            // 0-1
        
        // Extract scale components more safely
        uint8_t scale_byte = Data[offset++];
        double scale = 0.01 + (scale_byte / 255.0) * 0.99; // Scale between 0.01 and 1.0
        int64_t zero_point = Data[offset++] % 128;         // 0-127 for quint8
        
        // Create input tensor from remaining data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // Ensure the tensor is 4D (N, C, H, W) for MaxPool2d
        // First flatten to 1D then reshape
        input_tensor = input_tensor.flatten();
        int64_t total_elements = input_tensor.numel();
        
        if (total_elements < 16) {
            // Need minimum elements for a small 4D tensor
            return 0;
        }
        
        // Calculate reasonable 4D dimensions
        int64_t batch = 1;
        int64_t channels = 1;
        int64_t min_spatial = kernel_size + (kernel_size - 1) * (dilation - 1);
        if (min_spatial < 2) min_spatial = 2;
        
        // Try to create at least min_spatial x min_spatial spatial dims
        int64_t spatial_elements = total_elements / (batch * channels);
        int64_t height = static_cast<int64_t>(std::sqrt(static_cast<double>(spatial_elements)));
        if (height < min_spatial) height = min_spatial;
        int64_t width = spatial_elements / height;
        if (width < min_spatial) width = min_spatial;
        
        // Recalculate to fit available elements
        int64_t needed = batch * channels * height * width;
        if (needed > total_elements) {
            // Use minimum viable size
            height = min_spatial;
            width = min_spatial;
            needed = batch * channels * height * width;
            if (needed > total_elements) {
                return 0;
            }
        }
        
        // Take only the elements we need and reshape
        input_tensor = input_tensor.slice(0, 0, needed).reshape({batch, channels, height, width});
        
        // Convert to float for quantization
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Clamp values to reasonable range for quantization
        input_tensor = input_tensor.clamp(-100.0, 100.0);
        
        // Quantize the input tensor
        torch::Tensor quantized_input;
        try {
            quantized_input = torch::quantize_per_tensor(
                input_tensor, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            // Quantization failed, try with default tensor
            auto simple_tensor = torch::rand({1, 1, 8, 8}, torch::kFloat);
            quantized_input = torch::quantize_per_tensor(
                simple_tensor, 0.1, 0, torch::kQUInt8);
        }
        
        // Ensure padding doesn't exceed kernel bounds
        if (padding >= kernel_size) {
            padding = kernel_size - 1;
        }
        
        // Apply quantized max_pool2d
        torch::Tensor output;
        try {
            output = torch::max_pool2d(
                quantized_input,
                {kernel_size, kernel_size},
                {stride, stride},
                {padding, padding},
                {dilation, dilation},
                ceil_mode);
            
            // Dequantize output to verify the operation completed
            torch::Tensor dequantized_output = output.dequantize();
            
            // Access a value to ensure computation happens
            if (dequantized_output.numel() > 0) {
                volatile float val = dequantized_output.flatten()[0].item<float>();
                (void)val;
            }
        } catch (...) {
            // Inner catch for expected shape/parameter failures - silent
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}