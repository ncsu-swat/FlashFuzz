#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 3 bytes left for configuration
        if (Size - offset < 3) {
            return 0;
        }
        
        // Extract parameters for the quantized convolution
        uint8_t in_channels = Data[offset++] % 16 + 1;  // 1-16 input channels
        uint8_t out_channels = Data[offset++] % 16 + 1; // 1-16 output channels
        uint8_t kernel_size = Data[offset++] % 5 + 1;   // 1-5 kernel size
        
        // Determine stride, padding, dilation, groups based on remaining data
        int stride = 1;
        int padding = 0;
        int dilation = 1;
        int groups = 1;
        
        if (Size - offset >= 4) {
            stride = Data[offset++] % 3 + 1;     // 1-3 stride
            padding = Data[offset++] % 3;        // 0-2 padding
            dilation = Data[offset++] % 2 + 1;   // 1-2 dilation
            groups = Data[offset++] % in_channels + 1; // 1-in_channels groups
            
            // Ensure groups divides in_channels
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (Size - offset >= 9) {
            // Extract scale (as a double between 0.01 and 10.0)
            double scale_raw;
            std::memcpy(&scale_raw, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale_raw);
            scale = std::max(0.01, std::min(10.0, scale));
            
            // Extract zero_point (as an int64_t between -128 and 127)
            int64_t zero_point_raw;
            std::memcpy(&zero_point_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            zero_point = zero_point_raw % 256;
            if (zero_point > 127) zero_point -= 256;
        }
        
        // Create quantized tensors
        torch::Tensor weight;
        torch::Tensor bias;
        
        // Create weight tensor
        if (input.dim() >= 3) {
            // For Conv1d
            if (input.dim() == 3) {
                weight = torch::randn({out_channels, in_channels / groups, kernel_size});
            }
            // For Conv2d
            else if (input.dim() == 4) {
                weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size});
            }
            // For Conv3d
            else if (input.dim() == 5) {
                weight = torch::randn({out_channels, in_channels / groups, kernel_size, kernel_size, kernel_size});
            }
            
            // Create bias tensor
            bias = torch::randn({out_channels});
            
            // Quantize the input tensor
            auto q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQUInt8);
            
            // Create quantized convolution module based on input dimensions
            if (input.dim() == 3) {
                // Conv1d - use functional approach
                auto q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
                auto q_bias = torch::quantize_per_tensor(bias, scale * scale, 0, torch::kQInt32);
                
                auto output = torch::conv1d(q_input, q_weight, q_bias, stride, padding, dilation, groups);
                
                // Dequantize for further operations if needed
                auto dq_output = output.dequantize();
            }
            else if (input.dim() == 4) {
                // Conv2d - use functional approach
                auto q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
                auto q_bias = torch::quantize_per_tensor(bias, scale * scale, 0, torch::kQInt32);
                
                auto output = torch::conv2d(q_input, q_weight, q_bias, stride, padding, dilation, groups);
                
                // Dequantize for further operations if needed
                auto dq_output = output.dequantize();
            }
            else if (input.dim() == 5) {
                // Conv3d - use functional approach
                auto q_weight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
                auto q_bias = torch::quantize_per_tensor(bias, scale * scale, 0, torch::kQInt32);
                
                auto output = torch::conv3d(q_input, q_weight, q_bias, stride, padding, dilation, groups);
                
                // Dequantize for further operations if needed
                auto dq_output = output.dequantize();
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