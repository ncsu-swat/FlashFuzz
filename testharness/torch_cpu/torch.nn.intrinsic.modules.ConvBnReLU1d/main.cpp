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
        
        // Ensure input has at least 3 dimensions for ConvBnReLU1d (N, C, L)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvBnReLU1d from the remaining data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0;
        uint8_t stride = 1, padding = 0, dilation = 1, groups = 1;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
            stride = Data[offset++] % 3 + 1;       // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = Data[offset++] % 2 + 1;     // 1-2 dilation
            groups = Data[offset++] % 2 + 1;       // 1-2 groups
            
            // Ensure groups divides in_channels
            if (in_channels % groups != 0) {
                in_channels = groups;
            }
        }
        
        // Ensure input shape matches in_channels
        auto input_sizes = input.sizes().vec();
        if (input_sizes[1] != in_channels) {
            input = input.reshape({input_sizes[0], in_channels, -1});
        }
        
        // Create Conv1d module
        auto conv_options = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                            .stride(stride)
                                            .padding(padding)
                                            .dilation(dilation)
                                            .groups(groups)
                                            .bias(bias);
        torch::nn::Conv1d conv(conv_options);
        
        // Create BatchNorm1d module
        auto bn_options = torch::nn::BatchNorm1dOptions(out_channels);
        torch::nn::BatchNorm1d bn(bn_options);
        
        // Create ReLU module
        torch::nn::ReLU relu;
        
        // Initialize weights with data from the fuzzer input
        if (offset < Size) {
            auto weight_data = fuzzer_utils::createTensor(Data, Size, offset);
            if (weight_data.numel() > 0) {
                // Reshape weight data to match expected shape for Conv1d weights
                if (weight_data.numel() != conv->weight.numel()) {
                    weight_data = weight_data.reshape_as(conv->weight);
                }
                conv->weight.copy_(weight_data);
            }
        }
        
        // Initialize bias with data from the fuzzer input if there's still data left
        if (bias && offset < Size) {
            auto bias_data = fuzzer_utils::createTensor(Data, Size, offset);
            if (bias_data.numel() > 0) {
                // Reshape bias data to match expected shape
                if (bias_data.numel() != conv->bias.numel()) {
                    bias_data = bias_data.reshape_as(conv->bias);
                }
                conv->bias.copy_(bias_data);
            }
        }
        
        // Initialize BatchNorm parameters if there's still data left
        if (offset < Size) {
            auto bn_weight_data = fuzzer_utils::createTensor(Data, Size, offset);
            if (bn_weight_data.numel() > 0) {
                if (bn_weight_data.numel() != bn->weight.numel()) {
                    bn_weight_data = bn_weight_data.reshape_as(bn->weight);
                }
                bn->weight.copy_(bn_weight_data);
            }
        }
        
        if (offset < Size) {
            auto bn_bias_data = fuzzer_utils::createTensor(Data, Size, offset);
            if (bn_bias_data.numel() > 0) {
                if (bn_bias_data.numel() != bn->bias.numel()) {
                    bn_bias_data = bn_bias_data.reshape_as(bn->bias);
                }
                bn->bias.copy_(bn_bias_data);
            }
        }
        
        // Apply the ConvBnReLU1d operation (manually chaining the operations)
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor bn_output = bn->forward(conv_output);
        torch::Tensor output = relu->forward(bn_output);
        
        // Ensure output is valid
        if (output.isnan().any().item<bool>() || output.isinf().any().item<bool>()) {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
