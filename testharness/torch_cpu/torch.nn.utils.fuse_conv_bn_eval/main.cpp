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
        
        // Create a convolutional module
        int64_t in_channels = (Data[0] % 8) + 1;
        int64_t out_channels = (Data[1] % 8) + 1;
        int64_t kernel_size = (Data[2] % 5) + 1;
        int64_t stride = (Data[3] % 3) + 1;
        int64_t padding = Data[4] % 3;
        int64_t dilation = (Data[5] % 2) + 1;
        int64_t groups = std::max(1L, std::gcd(in_channels, out_channels));
        bool bias = Data[6] % 2 == 0;
        
        offset = 7;
        
        torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
        
        auto conv = torch::nn::Conv2d(conv_options);
        
        // Create batch normalization module
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset < Size) {
            eps = static_cast<double>(Data[offset++]) / 255.0 * 1e-3 + 1e-6;
        }
        
        if (offset < Size) {
            momentum = static_cast<double>(Data[offset++]) / 255.0;
        }
        
        torch::nn::BatchNorm2dOptions bn_options = torch::nn::BatchNorm2dOptions(out_channels)
            .eps(eps)
            .momentum(momentum)
            .affine(true)
            .track_running_stats(true);
        
        auto bn = torch::nn::BatchNorm2d(bn_options);
        
        // Set modules to eval mode
        conv->eval();
        bn->eval();
        
        // Create input tensor for testing
        std::vector<int64_t> input_shape;
        int64_t batch_size = 1;
        
        if (offset < Size) {
            batch_size = (Data[offset++] % 4) + 1;
        }
        
        input_shape.push_back(batch_size);
        input_shape.push_back(in_channels);
        
        int64_t height = 8;
        int64_t width = 8;
        
        if (offset + 1 < Size) {
            height = (Data[offset++] % 16) + 1;
            width = (Data[offset++] % 16) + 1;
        }
        
        input_shape.push_back(height);
        input_shape.push_back(width);
        
        torch::Tensor input;
        try {
            input = torch::rand(input_shape);
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Initialize running stats for batch norm
        torch::Tensor running_mean = torch::zeros({out_channels});
        torch::Tensor running_var = torch::ones({out_channels});
        
        // Assign running stats to batch norm module
        bn->running_mean = running_mean;
        bn->running_var = running_var;
        
        // Try different edge cases for weight and bias
        if (offset < Size) {
            uint8_t weight_case = Data[offset++] % 4;
            
            switch (weight_case) {
                case 0:
                    // Normal weights
                    break;
                case 1:
                    // Very small weights
                    conv->weight = conv->weight * 1e-10;
                    bn->weight = bn->weight * 1e-10;
                    break;
                case 2:
                    // Very large weights
                    conv->weight = conv->weight * 1e10;
                    bn->weight = bn->weight * 1e10;
                    break;
                case 3:
                    // Zero weights
                    conv->weight = torch::zeros_like(conv->weight);
                    bn->weight = torch::zeros_like(bn->weight);
                    break;
            }
        }
        
        if (offset < Size && bias) {
            uint8_t bias_case = Data[offset++] % 4;
            
            switch (bias_case) {
                case 0:
                    // Normal bias
                    break;
                case 1:
                    // Very small bias
                    conv->bias = conv->bias * 1e-10;
                    bn->bias = bn->bias * 1e-10;
                    break;
                case 2:
                    // Very large bias
                    conv->bias = conv->bias * 1e10;
                    bn->bias = bn->bias * 1e10;
                    break;
                case 3:
                    // Zero bias
                    conv->bias = torch::zeros_like(conv->bias);
                    bn->bias = torch::zeros_like(bn->bias);
                    break;
            }
        }
        
        // Manual fusion of conv and bn in eval mode
        try {
            // Get conv parameters
            torch::Tensor conv_weight = conv->weight;
            torch::Tensor conv_bias = bias ? conv->bias : torch::zeros({out_channels});
            
            // Get bn parameters
            torch::Tensor bn_weight = bn->weight;
            torch::Tensor bn_bias = bn->bias;
            torch::Tensor bn_running_mean = bn->running_mean;
            torch::Tensor bn_running_var = bn->running_var;
            
            // Compute fused parameters
            torch::Tensor scale = bn_weight / torch::sqrt(bn_running_var + eps);
            torch::Tensor fused_bias = bn_bias - bn_running_mean * scale + conv_bias * scale;
            
            // Reshape scale for broadcasting with conv weight
            std::vector<int64_t> scale_shape(conv_weight.dim(), 1);
            scale_shape[0] = out_channels;
            torch::Tensor scale_reshaped = scale.view(scale_shape);
            
            torch::Tensor fused_weight = conv_weight * scale_reshaped;
            
            // Create fused conv module
            torch::nn::Conv2dOptions fused_options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(true);
            
            auto fused_conv = torch::nn::Conv2d(fused_options);
            fused_conv->weight = fused_weight;
            fused_conv->bias = fused_bias;
            fused_conv->eval();
            
            // Verify the fused module works by running inference
            torch::Tensor output = fused_conv->forward(input);
            
            // Compare with original modules
            torch::Tensor expected_output = bn->forward(conv->forward(input));
            
            // Check if outputs are close
            bool close = torch::allclose(output, expected_output, 1e-4, 1e-5);
            if (!close) {
                // This is not an error, just an interesting case to investigate
                return 1;
            }
        } catch (const std::exception& e) {
            // Catch any exceptions from the fusion operation
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