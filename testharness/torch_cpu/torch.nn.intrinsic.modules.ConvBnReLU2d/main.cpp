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
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for Conv2d
        if (input.dim() < 4) {
            input = input.reshape({1, 1, 
                                  input.numel() > 1 ? 2 : 1, 
                                  input.numel() > 2 ? input.numel() / 2 : 1});
        }
        
        // Extract parameters for ConvBnReLU2d from the remaining data
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
            groups = Data[offset++] % std::min(in_channels, out_channels) + 1; // 1-min(in,out) groups
            
            // Ensure in_channels is divisible by groups
            in_channels = (in_channels / groups) * groups;
            if (in_channels == 0) in_channels = groups;
            
            // Ensure out_channels is divisible by groups
            out_channels = (out_channels / groups) * groups;
            if (out_channels == 0) out_channels = groups;
            
            // Get bias flag if there's more data
            if (offset < Size) {
                bias = Data[offset++] % 2 == 0;
            }
        } else {
            // Default values if not enough data
            in_channels = 3;
            out_channels = 6;
            kernel_size = 3;
        }
        
        // Ensure input has the right number of channels
        if (input.size(1) != in_channels) {
            input = input.expand({input.size(0), in_channels, input.size(2), input.size(3)});
        }
        
        // Create momentum and eps parameters for BatchNorm
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset + 2 <= Size) {
            // Use remaining data to influence momentum and eps
            uint16_t momentum_raw = 0;
            std::memcpy(&momentum_raw, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            
            // Convert to a reasonable momentum value (0.01 to 0.99)
            momentum = 0.01 + (momentum_raw % 99) / 100.0;
            
            if (offset + 2 <= Size) {
                uint16_t eps_raw = 0;
                std::memcpy(&eps_raw, Data + offset, sizeof(uint16_t));
                offset += sizeof(uint16_t);
                
                // Convert to a reasonable eps value (1e-6 to 1e-3)
                eps = std::pow(10, -6 + (eps_raw % 4));
            }
        }
        
        // Create Conv2d, BatchNorm2d, and ReLU modules separately
        auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                     .stride(stride)
                                     .padding(padding)
                                     .dilation(dilation)
                                     .groups(groups)
                                     .bias(bias));
        
        auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)
                                        .momentum(momentum)
                                        .eps(eps));
        
        auto relu = torch::nn::ReLU();
        
        // Apply the modules sequentially to simulate ConvBnReLU2d
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor bn_output = bn->forward(conv_output);
        torch::Tensor output = relu->forward(bn_output);
        
        // Perform some basic checks on the output
        if (output.numel() == 0 || !output.defined() || output.isnan().any().item<bool>() || 
            output.isinf().any().item<bool>()) {
            throw std::runtime_error("Invalid output tensor");
        }
        
        // Test the modules in training and eval modes
        conv->train();
        bn->train();
        relu->train();
        torch::Tensor conv_train = conv->forward(input);
        torch::Tensor bn_train = bn->forward(conv_train);
        torch::Tensor output_train = relu->forward(bn_train);
        
        conv->eval();
        bn->eval();
        relu->eval();
        torch::Tensor conv_eval = conv->forward(input);
        torch::Tensor bn_eval = bn->forward(conv_eval);
        torch::Tensor output_eval = relu->forward(bn_eval);
        
        // Test with different input shapes if possible
        if (Size > offset + 10) {
            // Create a new input with different spatial dimensions
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Reshape to valid dimensions for Conv2d
            if (input2.dim() < 4) {
                input2 = input2.reshape({1, in_channels, 
                                       input2.numel() > in_channels ? 3 : 1, 
                                       input2.numel() > in_channels * 3 ? 3 : 1});
            } else if (input2.size(1) != in_channels) {
                input2 = input2.expand({input2.size(0), in_channels, input2.size(2), input2.size(3)});
            }
            
            // Forward pass with the new input
            torch::Tensor conv2 = conv->forward(input2);
            torch::Tensor bn2 = bn->forward(conv2);
            torch::Tensor output2 = relu->forward(bn2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
