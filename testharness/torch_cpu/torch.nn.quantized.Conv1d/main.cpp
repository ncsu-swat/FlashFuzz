#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor has at least 3 dimensions for Conv1d (batch, channels, length)
        if (input_tensor.dim() < 3) {
            input_tensor = input_tensor.reshape({1, 1, input_tensor.numel()});
        }
        
        // Extract parameters for Conv1d from the remaining data
        int64_t in_channels = input_tensor.size(1);
        
        // Get out_channels from data
        int64_t out_channels = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 16 + 1; // Limit to reasonable range
        }
        
        // Get kernel_size from data
        int64_t kernel_size = 3;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 7 + 1; // Limit to reasonable range
        }
        
        // Get stride from data
        int64_t stride = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 3 + 1; // Limit to reasonable range
        }
        
        // Get padding from data
        int64_t padding = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 3; // Limit to reasonable range
        }
        
        // Get dilation from data
        int64_t dilation = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 2 + 1; // Limit to reasonable range
        }
        
        // Get groups from data
        int64_t groups = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % std::max(in_channels, static_cast<int64_t>(1)) + 1;
            if (groups > in_channels) groups = in_channels;
            if (in_channels % groups != 0) groups = 1; // Ensure in_channels is divisible by groups
        }
        
        // Get bias flag from data
        bool use_bias = true;
        if (offset < Size) {
            use_bias = Data[offset++] & 0x1;
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale);
            if (scale < 1e-6) scale = 1e-6;
            if (scale > 1.0) scale = 1.0;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            zero_point = zero_point % 256; // Limit to reasonable range for int8
        }
        
        // Create quantized input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(
                input_tensor.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQInt8
            );
        } catch (const std::exception& e) {
            // If quantization fails, create a default quantized tensor
            q_input = torch::quantize_per_tensor(
                torch::ones({1, in_channels, 10}), 
                1.0, 
                0, 
                torch::kQInt8
            );
        }
        
        // Create weight tensor for Conv1d
        torch::Tensor weight = torch::randn({out_channels, in_channels / groups, kernel_size});
        
        // Quantize weight tensor
        torch::Tensor q_weight = torch::quantize_per_tensor(
            weight, 
            scale, 
            zero_point, 
            torch::kQInt8
        );
        
        // Create bias tensor if needed
        torch::Tensor bias_tensor;
        if (use_bias) {
            bias_tensor = torch::randn({out_channels});
        }
        
        // Use functional quantized conv1d directly
        torch::Tensor output = torch::nn::functional::conv1d(
            q_input,
            q_weight,
            use_bias ? torch::optional<torch::Tensor>(bias_tensor) : torch::nullopt,
            torch::nn::functional::Conv1dFuncOptions()
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
        );
        
        // Dequantize output for further processing if needed
        torch::Tensor dequantized_output = output.dequantize();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}