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
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for ConvReLU1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvReLU1d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1 + (offset < Size ? Data[offset++] % 8 : 1);
        int64_t kernel_size = 1 + (offset < Size ? Data[offset++] % 5 : 1);
        int64_t stride = 1 + (offset < Size ? Data[offset++] % 3 : 1);
        int64_t padding = (offset < Size ? Data[offset++] % 3 : 0);
        int64_t dilation = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t groups = 1;
        bool bias = (offset < Size ? (Data[offset++] % 2 == 0) : true);
        
        // Ensure groups divides in_channels
        if (in_channels > 1 && offset < Size) {
            groups = 1 + (Data[offset++] % in_channels);
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        // Create weight tensor
        std::vector<int64_t> weight_shape = {out_channels, in_channels / groups, kernel_size};
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
            if (weight.dim() != 3 || weight.size(0) != out_channels || 
                weight.size(1) != in_channels / groups || weight.size(2) != kernel_size) {
                weight = torch::ones(weight_shape, torch::kFloat);
            }
        } else {
            weight = torch::ones(weight_shape, torch::kFloat);
        }
        
        // Create bias tensor if needed
        torch::Tensor bias_tensor;
        if (bias) {
            if (offset < Size) {
                bias_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (bias_tensor.dim() != 1 || bias_tensor.size(0) != out_channels) {
                    bias_tensor = torch::zeros({out_channels}, torch::kFloat);
                }
            } else {
                bias_tensor = torch::zeros({out_channels}, torch::kFloat);
            }
        }
        
        // Quantize input tensor
        torch::Tensor q_input = torch::quantize_per_tensor(
            input.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
        
        // Quantize weight tensor
        torch::Tensor q_weight = torch::quantize_per_tensor(
            weight.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
        
        // Use functional API for quantized conv1d + relu
        torch::Tensor output = torch::nn::functional::conv1d(
            q_input, q_weight, 
            bias ? c10::optional<torch::Tensor>(bias_tensor) : c10::nullopt,
            stride, padding, dilation, groups);
        
        // Apply ReLU
        output = torch::relu(output);
        
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
