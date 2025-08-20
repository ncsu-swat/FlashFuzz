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
        
        // Create input tensors for intrinsic modules
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor for operations that need multiple inputs
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create a simple one
            weight = torch::ones({3, 3});
        }
        
        // Get some configuration parameters from the input data
        int in_channels = 3;
        int out_channels = 3;
        int kernel_size = 3;
        int stride = 1;
        int padding = 1;
        
        if (offset + 5 <= Size) {
            in_channels = 1 + (Data[offset++] % 8);
            out_channels = 1 + (Data[offset++] % 8);
            kernel_size = 1 + (Data[offset++] % 5);
            stride = 1 + (Data[offset++] % 3);
            padding = Data[offset++] % 3;
        }
        
        // Test regular conv2d with batch norm and relu as separate operations
        try {
            auto conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .bias(true));
            
            auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
            
            // Reshape input if needed to match conv requirements
            torch::Tensor conv_input = input;
            if (input.dim() < 4) {
                conv_input = input.reshape({1, in_channels, 
                                           std::max<int64_t>(1, input.size(0)), 
                                           std::max<int64_t>(1, input.size(0))});
            } else if (input.size(1) != in_channels) {
                conv_input = conv_input.repeat({1, in_channels, 1, 1});
                conv_input = conv_input.slice(2, 0, std::max<int64_t>(1, input.size(2)));
                conv_input = conv_input.slice(3, 0, std::max<int64_t>(1, input.size(3)));
            }
            
            auto conv_output = conv->forward(conv_input);
            auto bn_output = bn->forward(conv_output);
            auto relu_output = torch::relu(bn_output);
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        try {
            // Conv2d with ReLU
            auto conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .bias(true));
            
            // Reshape input if needed
            torch::Tensor conv_input = input;
            if (input.dim() < 4) {
                conv_input = input.reshape({1, in_channels, 
                                           std::max<int64_t>(1, input.size(0)), 
                                           std::max<int64_t>(1, input.size(0))});
            } else if (input.size(1) != in_channels) {
                conv_input = conv_input.repeat({1, in_channels, 1, 1});
                conv_input = conv_input.slice(2, 0, std::max<int64_t>(1, input.size(2)));
                conv_input = conv_input.slice(3, 0, std::max<int64_t>(1, input.size(3)));
            }
            
            auto conv_output = conv->forward(conv_input);
            auto relu_output = torch::relu(conv_output);
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        try {
            // Conv3d with ReLU
            auto conv3d = torch::nn::Conv3d(
                torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .bias(true));
            
            // Reshape input if needed
            torch::Tensor conv_input = input;
            if (input.dim() < 5) {
                conv_input = input.reshape({1, in_channels, 
                                           std::max<int64_t>(1, input.size(0)), 
                                           std::max<int64_t>(1, input.size(0)),
                                           std::max<int64_t>(1, input.size(0))});
            } else if (input.size(1) != in_channels) {
                conv_input = conv_input.repeat({1, in_channels, 1, 1, 1});
                conv_input = conv_input.slice(2, 0, std::max<int64_t>(1, input.size(2)));
                conv_input = conv_input.slice(3, 0, std::max<int64_t>(1, input.size(3)));
                conv_input = conv_input.slice(4, 0, std::max<int64_t>(1, input.size(4)));
            }
            
            auto conv_output = conv3d->forward(conv_input);
            auto relu_output = torch::relu(conv_output);
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        try {
            // Linear with ReLU
            auto linear = torch::nn::Linear(
                torch::nn::LinearOptions(in_channels, out_channels).bias(true));
            
            // Reshape input if needed
            torch::Tensor linear_input = input;
            if (input.dim() == 0) {
                linear_input = input.reshape({1, in_channels});
            } else if (input.dim() == 1) {
                if (input.size(0) != in_channels) {
                    linear_input = linear_input.repeat({in_channels}).slice(0, 0, in_channels);
                }
                linear_input = linear_input.reshape({1, in_channels});
            } else {
                // For higher dimensions, flatten the last dimension to match in_channels
                auto sizes = input.sizes().vec();
                sizes.back() = in_channels;
                linear_input = linear_input.reshape(sizes);
            }
            
            auto linear_output = linear->forward(linear_input);
            auto relu_output = torch::relu(linear_output);
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // Test functional operations
        try {
            // Try add and relu operation
            if (input.dim() > 0 && weight.dim() > 0) {
                // Make sure tensors have compatible shapes for addition
                torch::Tensor a = input;
                torch::Tensor b = weight;
                
                // Reshape if necessary
                if (a.dim() != b.dim()) {
                    if (a.dim() < b.dim()) {
                        std::vector<int64_t> new_shape(b.dim(), 1);
                        for (int i = 0; i < a.dim(); i++) {
                            new_shape[i] = a.size(i);
                        }
                        a = a.reshape(new_shape);
                    } else {
                        std::vector<int64_t> new_shape(a.dim(), 1);
                        for (int i = 0; i < b.dim(); i++) {
                            new_shape[i] = b.size(i);
                        }
                        b = b.reshape(new_shape);
                    }
                }
                
                // Try the add and relu operation
                auto add_output = torch::add(a, b);
                auto add_relu_output = torch::relu(add_output);
            }
        } catch (const std::exception& e) {
            // Continue with other tests
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}