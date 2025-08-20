#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get some configuration parameters from the input data
        uint8_t config_byte = 0;
        if (offset < Size) {
            config_byte = Data[offset++];
        }
        
        // Create a quantized model using torch::nn modules
        torch::nn::Linear linear = torch::nn::Linear(input.size(-1), input.size(-1));
        
        // Try different modules based on the config byte
        switch (config_byte % 4) {
            case 0: {
                // Test Linear with ReLU
                auto output = torch::relu(linear->forward(input));
                break;
            }
            case 1: {
                // Test Conv2d with ReLU if input has enough dimensions
                if (input.dim() >= 4) {
                    int64_t in_channels = input.size(1);
                    int64_t out_channels = in_channels;
                    
                    torch::nn::Conv2d conv = torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1));
                    
                    auto output = torch::relu(conv->forward(input));
                }
                break;
            }
            case 2: {
                // Test Conv2d with BatchNorm if input has enough dimensions
                if (input.dim() >= 4) {
                    int64_t in_channels = input.size(1);
                    int64_t out_channels = in_channels;
                    
                    torch::nn::Conv2d conv = torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1));
                    
                    torch::nn::BatchNorm2d bn = torch::nn::BatchNorm2d(out_channels);
                    
                    auto conv_output = conv->forward(input);
                    auto output = bn->forward(conv_output);
                }
                break;
            }
            case 3: {
                // Test Conv2d with BatchNorm and ReLU if input has enough dimensions
                if (input.dim() >= 4) {
                    int64_t in_channels = input.size(1);
                    int64_t out_channels = in_channels;
                    
                    torch::nn::Conv2d conv = torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1));
                    
                    torch::nn::BatchNorm2d bn = torch::nn::BatchNorm2d(out_channels);
                    
                    auto conv_output = conv->forward(input);
                    auto bn_output = bn->forward(conv_output);
                    auto output = torch::relu(bn_output);
                }
                break;
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