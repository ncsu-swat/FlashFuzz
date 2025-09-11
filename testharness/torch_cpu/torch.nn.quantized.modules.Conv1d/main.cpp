#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions for Conv1d (N, C, L)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for Conv1d from the remaining data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0, stride = 0, padding = 0, dilation = 0, groups = 0;
        bool bias = false;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;       // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = Data[offset++] % 2 + 1;     // 1-2 dilation
            groups = Data[offset++] % 2 + 1;       // 1-2 groups
            
            // Ensure groups compatibility
            if (in_channels % groups != 0 || out_channels % groups != 0) {
                groups = 1;
            }
            
            // Get bias flag if data available
            if (offset < Size) {
                bias = Data[offset++] & 1;
            }
        } else {
            // Default values if not enough data
            in_channels = 1;
            out_channels = 1;
            kernel_size = 1;
            stride = 1;
            padding = 0;
            dilation = 1;
            groups = 1;
        }
        
        // Ensure input has correct number of channels
        if (input.size(1) != in_channels) {
            input = input.reshape({input.size(0), in_channels, -1});
        }
        
        // Create scale and zero_point for quantization
        double scale_input = 0.1;
        int64_t zero_point_input = 10;
        
        if (offset + 8 <= Size) {
            // Extract scale and zero_point from data if available
            double scale_raw;
            std::memcpy(&scale_raw, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive and reasonable
            scale_input = std::abs(scale_raw);
            if (scale_input < 1e-6) scale_input = 0.1;
            if (scale_input > 1e6) scale_input = 1.0;
            
            // Extract zero point
            zero_point_input = static_cast<int64_t>(Data[offset++]) % 256;
        }
        
        // Quantize the input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                input.to(torch::kFloat), 
                scale_input, 
                zero_point_input, 
                torch::kQUInt8
            );
        } catch (...) {
            // If quantization fails, create a default quantized tensor
            input = torch::ones({1, in_channels, 10}, torch::kFloat);
            q_input = torch::quantize_per_tensor(input, scale_input, zero_point_input, torch::kQUInt8);
        }
        
        // Create regular Conv1d module and simulate quantized behavior
        torch::nn::Conv1dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        // Create the Conv1d module
        torch::nn::Conv1d conv(options);
        
        // Dequantize input for regular conv1d
        torch::Tensor dequant_input = q_input.dequantize();
        
        // Forward pass
        torch::Tensor output = conv(dequant_input);
        
        // Quantize output to simulate quantized conv behavior
        torch::Tensor q_output = torch::quantize_per_tensor(
            output, 
            0.1, 
            0, 
            torch::kQUInt8
        );
        
        // Try to access some properties of the output to ensure it's valid
        auto sizes = q_output.sizes();
        auto dtype = q_output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        return 0; // keep the input
    }
    return 0; // keep the input
}
