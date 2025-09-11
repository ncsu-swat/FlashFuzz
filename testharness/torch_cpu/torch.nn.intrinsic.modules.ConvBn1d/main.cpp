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
        
        // Ensure input has at least 3 dimensions (N, C, L) for Conv1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for Conv1d from the remaining data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        bool bias = true;
        
        // Parse more parameters if data available
        if (offset + 7 <= Size) {
            out_channels = (Data[offset] % 8) + 1;
            kernel_size = (Data[offset + 1] % 5) + 1;
            stride = (Data[offset + 2] % 3) + 1;
            padding = Data[offset + 3] % 3;
            dilation = (Data[offset + 4] % 2) + 1;
            bias = Data[offset + 5] % 2 == 0;
            offset += 7;
        }
        
        // Create Conv1d module
        auto conv_options = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                            .stride(stride)
                                            .padding(padding)
                                            .dilation(dilation)
                                            .bias(bias);
        torch::nn::Conv1d conv(conv_options);
        
        // Create BatchNorm1d module
        auto bn_options = torch::nn::BatchNorm1dOptions(out_channels);
        torch::nn::BatchNorm1d bn(bn_options);
        
        // Set to evaluation mode to test both training and eval paths
        if (offset < Size && Data[offset] % 2 == 0) {
            conv->eval();
            bn->eval();
        }
        
        // Apply the modules sequentially (simulating fused ConvBn1d)
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = bn->forward(conv_output);
        
        // Try to access some properties and methods of the modules
        if (offset < Size) {
            if (Data[offset] % 5 == 0) {
                auto weight = conv->weight;
            }
            if (Data[offset] % 5 == 1) {
                auto bias_tensor = conv->bias;
            }
            if (Data[offset] % 5 == 2) {
                auto running_mean = bn->running_mean;
            }
            if (Data[offset] % 5 == 3) {
                auto running_var = bn->running_var;
            }
            if (Data[offset] % 5 == 4) {
                auto bn_weight = bn->weight;
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
