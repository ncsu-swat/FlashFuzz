#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for fused modules
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get some configuration parameters from the input data
        uint8_t in_features = 0;
        uint8_t out_features = 0;
        bool bias = true;
        
        if (offset + 2 < Size) {
            in_features = Data[offset++] % 32 + 1;  // Ensure non-zero
            out_features = Data[offset++] % 32 + 1; // Ensure non-zero
            if (offset < Size) {
                bias = Data[offset++] & 1;  // 0 or 1
            }
        } else {
            // Default values if not enough data
            in_features = 4;
            out_features = 4;
        }
        
        // Reshape input tensor if needed to match expected input shape for Linear
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                input = input.reshape({1, in_features});
            } else {
                input = input.reshape({input.size(0), in_features});
            }
        } else {
            input = input.reshape({input.size(0), in_features});
        }
        
        // Convert to float for compatibility with most operations
        input = input.to(torch::kFloat);
        
        // Test basic fused operations using functional approach
        
        // 1. Test Linear + ReLU fusion manually
        torch::nn::LinearOptions linear_options(in_features, out_features);
        linear_options.bias(bias);
        
        auto linear = torch::nn::Linear(linear_options);
        auto linear_output = linear(input);
        auto linear_relu_output = torch::relu(linear_output);
        
        // 2. Test Conv2d + ReLU fusion manually
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 1;
        
        // Reshape input for Conv2d if needed
        torch::Tensor conv_input;
        if (input.dim() < 4) {
            // Create a 4D tensor with shape [batch_size, channels, height, width]
            int64_t batch_size = input.size(0);
            int64_t channels = 3;  // Standard RGB channels
            int64_t height = 32;
            int64_t width = 32;
            
            // Reshape or create new tensor
            conv_input = torch::ones({batch_size, channels, height, width});
        } else {
            conv_input = input;
        }
        
        torch::nn::Conv2dOptions conv_options(conv_input.size(1), out_features, kernel_size);
        conv_options.stride(stride).padding(padding).bias(bias);
        
        auto conv = torch::nn::Conv2d(conv_options);
        auto conv_output = conv(conv_input);
        auto conv_relu_output = torch::relu(conv_output);
        
        // 3. Test Conv2d + BatchNorm2d fusion manually
        torch::nn::BatchNorm2dOptions bn_options(out_features);
        auto bn = torch::nn::BatchNorm2d(bn_options);
        auto conv_bn_output = bn(conv_output);
        
        // 4. Test Conv2d + BatchNorm2d + ReLU fusion manually
        auto conv_bn_relu_output = torch::relu(conv_bn_output);
        
        // 5. Test BatchNorm2d + ReLU fusion manually
        auto bn_relu_output = torch::relu(conv_bn_output);
        
        // Try to access some properties and methods
        auto weight = linear->weight;
        
        // Try to call some methods that might trigger edge cases
        linear->reset_parameters();
        linear->to(torch::kFloat16);
        linear->to(torch::kFloat);
        
        // Try serialization
        torch::serialize::OutputArchive output_archive;
        linear->save(output_archive);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}