#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get some configuration parameters from the input data
        uint8_t module_type = 0;
        if (offset < Size) {
            module_type = Data[offset++] % 3; // Choose between different module types
        }
        
        uint8_t in_channels = 3;
        if (offset < Size) {
            in_channels = 1 + (Data[offset++] % 32); // 1-32 input channels
        }
        
        uint8_t out_channels = 3;
        if (offset < Size) {
            out_channels = 1 + (Data[offset++] % 32); // 1-32 output channels
        }
        
        uint8_t kernel_size = 3;
        if (offset < Size) {
            kernel_size = 1 + (Data[offset++] % 7); // 1-7 kernel size
        }
        
        // Create modules and test them
        try {
            // Reshape input tensor to match expected input shape for conv modules
            // For conv modules, we need at least 3D tensor (N, C, H, W)
            if (input.dim() < 3) {
                std::vector<int64_t> new_shape = {1, in_channels, kernel_size};
                if (input.dim() < 2) {
                    new_shape.push_back(kernel_size);
                }
                input = input.reshape(new_shape);
            }
            
            // Make sure input has correct number of channels for conv operations
            if (input.size(1) != in_channels) {
                input = input.expand({input.size(0), in_channels, input.size(2), 
                                     input.dim() > 3 ? input.size(3) : kernel_size});
            }
            
            // Add a dimension if needed for 2D operations
            if (input.dim() == 3) {
                input = input.unsqueeze(-1);
            }
            
            // Test different modules based on module_type
            switch (module_type) {
                case 0: {
                    // Conv2d
                    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                          .stride(1).padding(kernel_size/2));
                    
                    // Forward pass
                    auto output = conv->forward(input);
                    break;
                }
                case 1: {
                    // Conv2d with ReLU
                    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                          .stride(1).padding(kernel_size/2));
                    torch::nn::ReLU relu;
                    
                    // Forward pass
                    auto conv_output = conv->forward(input);
                    auto output = relu->forward(conv_output);
                    break;
                }
                case 2: {
                    // Linear
                    // Reshape input for linear layer
                    auto batch_size = input.size(0);
                    auto flattened_input = input.reshape({batch_size, -1});
                    auto in_features = flattened_input.size(1);
                    
                    torch::nn::Linear linear(in_features, out_channels);
                    
                    // Forward pass
                    auto output = linear->forward(flattened_input);
                    break;
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and can be ignored
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
