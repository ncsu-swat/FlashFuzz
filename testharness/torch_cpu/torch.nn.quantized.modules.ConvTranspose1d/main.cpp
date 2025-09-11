#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Early return if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvTranspose1d
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0;
        uint8_t stride = 1, padding = 0, output_padding = 0, dilation = 1, groups = 1;
        
        if (offset + 8 <= Size) {
            in_channels = Data[offset++] % 8 + 1;
            out_channels = Data[offset++] % 8 + 1;
            kernel_size = Data[offset++] % 5 + 1;
            stride = Data[offset++] % 3 + 1;
            padding = Data[offset++] % 3;
            output_padding = Data[offset++] % 2;
            dilation = Data[offset++] % 2 + 1;
            groups = Data[offset++] % 2 + 1;
            
            // Ensure groups divides both in_channels and out_channels
            if (in_channels % groups != 0) {
                in_channels = groups;
            }
            if (out_channels % groups != 0) {
                out_channels = groups;
            }
        }
        
        // Reshape input to match expected dimensions for ConvTranspose1d
        int64_t batch_size = input.size(0);
        int64_t seq_len = input.size(input.dim() - 1);
        
        input = input.reshape({batch_size, in_channels, seq_len});
        
        // Create scale and zero_point for quantization
        double scale = 1.0 / 128.0;
        int64_t zero_point = 0;
        
        // Create quantized input tensor
        torch::Tensor q_input = torch::quantize_per_tensor(
            input.to(torch::kFloat), 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Create weight tensor
        std::vector<int64_t> weight_shape = {in_channels, out_channels / groups, kernel_size};
        torch::Tensor weight = torch::randn(weight_shape);
        
        // Create quantized weight
        torch::Tensor q_weight = torch::quantize_per_tensor(
            weight, 
            scale, 
            zero_point, 
            torch::kQUInt8
        );
        
        // Create bias tensor
        torch::Tensor bias = torch::randn({out_channels});
        
        // Use functional API for quantized conv_transpose1d
        try {
            torch::Tensor output = torch::conv_transpose1d(
                q_input,
                q_weight,
                bias,
                stride,
                padding,
                output_padding,
                groups,
                dilation
            );
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and not a fuzzer error
            return 0;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
