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
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for ConvBn1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvBn1d from the remaining data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0;
        int64_t stride = 1, padding = 0, dilation = 1, groups = 1;
        bool bias = true;
        
        if (offset + 3 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
        }
        
        if (offset + 4 <= Size) {
            stride = (Data[offset++] % 3) + 1;     // 1-3 stride
            padding = Data[offset++] % 3;          // 0-2 padding
            dilation = (Data[offset++] % 2) + 1;   // 1-2 dilation
            groups = std::gcd(in_channels, out_channels);
            if (groups > 1 && Data[offset++] % 2 == 0) {
                groups = 1; // Sometimes use grouped convolution
            }
            bias = Data[offset++] % 2 == 0;        // Random bias
        }
        
        // Ensure input shape is compatible with parameters
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t length = input.size(2);
        
        // Adjust input tensor if needed
        if (channels != in_channels) {
            input = input.reshape({batch_size, in_channels, -1});
            length = input.size(2);
        }
        
        // Create Conv1d options
        auto conv_options = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                            .stride(stride)
                                            .padding(padding)
                                            .dilation(dilation)
                                            .groups(groups)
                                            .bias(bias);
        
        // Create Conv1d module
        torch::nn::Conv1d conv(conv_options);
        
        // Create BatchNorm1d module
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(out_channels));
        
        // Set to train mode
        conv->train();
        bn->train();
        
        // Apply conv followed by batchnorm (simulating ConvBn1d behavior)
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Try inference mode too
        conv->eval();
        bn->eval();
        torch::Tensor conv_output_eval = conv->forward(input);
        torch::Tensor output_eval = bn->forward(conv_output_eval);
        
        // Try with different eps values
        if (offset < Size) {
            float eps = static_cast<float>(Data[offset++]) / 255.0f + 1e-6f;
            bn->options.eps(eps);
            torch::Tensor conv_output_eps = conv->forward(input);
            torch::Tensor output_eps = bn->forward(conv_output_eps);
        }
        
        // Try with different momentum values
        if (offset < Size) {
            float momentum = static_cast<float>(Data[offset++]) / 255.0f;
            bn->options.momentum(momentum);
            torch::Tensor conv_output_momentum = conv->forward(input);
            torch::Tensor output_momentum = bn->forward(conv_output_momentum);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
