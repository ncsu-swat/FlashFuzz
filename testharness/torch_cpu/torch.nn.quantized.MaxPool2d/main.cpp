#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 3 dimensions (N, C, H, W) for MaxPool2d
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // Quantize the input tensor
        float scale = 0.1f;
        int zero_point = 10;
        
        // Try to get scale and zero_point from the input data
        if (offset + sizeof(float) + sizeof(int) <= Size) {
            memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        // Ensure scale is positive and not too large
        scale = std::abs(scale);
        if (scale < 1e-6f) scale = 1e-6f;
        if (scale > 1e6f) scale = 1e6f;
        
        // Ensure zero_point is within valid range for quint8
        zero_point = std::max(0, std::min(zero_point, 255));
        
        // Quantize the tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto simple_tensor = torch::ones({1, 1, 4, 4}, options);
            q_input = torch::quantize_per_tensor(simple_tensor, 0.1, 10, torch::kQUInt8);
        }
        
        // Parse MaxPool2d parameters from the input data
        int kernel_size = 2;
        int stride = 1;
        int padding = 0;
        int dilation = 1;
        bool ceil_mode = false;
        
        if (offset + 5 * sizeof(int) <= Size) {
            memcpy(&kernel_size, Data + offset, sizeof(int));
            offset += sizeof(int);
            memcpy(&stride, Data + offset, sizeof(int));
            offset += sizeof(int);
            memcpy(&padding, Data + offset, sizeof(int));
            offset += sizeof(int);
            memcpy(&dilation, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // For ceil_mode, just use one byte
            ceil_mode = (Data[offset++] % 2 == 1);
        }
        
        // Ensure parameters are within reasonable ranges
        kernel_size = std::abs(kernel_size) % 5 + 1;  // 1 to 5
        stride = std::abs(stride) % 3 + 1;            // 1 to 3
        padding = std::abs(padding) % 3;              // 0 to 2
        dilation = std::abs(dilation) % 2 + 1;        // 1 to 2
        
        // Apply quantized max_pool2d directly using functional API
        torch::Tensor output = torch::max_pool2d(q_input, {kernel_size, kernel_size}, 
                                                 {stride, stride}, {padding, padding}, 
                                                 {dilation, dilation}, ceil_mode);
        
        // Try to access some properties of the output to ensure it's valid
        auto sizes = output.sizes();
        auto dtype = output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}