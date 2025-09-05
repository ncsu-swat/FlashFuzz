#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <torch/nn/intrinsic/qat.h>
#include <iostream>
#include <memory>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum bytes needed for basic parsing
        if (Size < 20) {
            return 0;  // Not enough data to construct meaningful inputs
        }

        // Parse convolution parameters from fuzzer input
        uint8_t in_channels_byte = (offset < Size) ? Data[offset++] : 3;
        uint8_t out_channels_byte = (offset < Size) ? Data[offset++] : 16;
        uint8_t kernel_size_byte = (offset < Size) ? Data[offset++] : 3;
        uint8_t stride_byte = (offset < Size) ? Data[offset++] : 1;
        uint8_t padding_byte = (offset < Size) ? Data[offset++] : 0;
        uint8_t dilation_byte = (offset < Size) ? Data[offset++] : 1;
        uint8_t groups_byte = (offset < Size) ? Data[offset++] : 1;
        bool bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        uint8_t padding_mode_selector = (offset < Size) ? Data[offset++] : 0;
        
        // BatchNorm parameters
        double eps_byte = (offset < Size) ? Data[offset++] : 100;
        double momentum_byte = (offset < Size) ? Data[offset++] : 10;
        bool affine = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        bool track_running_stats = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Quantization parameters
        uint8_t qscheme_selector = (offset < Size) ? Data[offset++] : 0;
        uint8_t reduce_range = (offset < Size) ? Data[offset++] : 0;
        
        // Convert bytes to meaningful values
        int64_t in_channels = 1 + (in_channels_byte % 32);
        int64_t out_channels = 1 + (out_channels_byte % 32);
        int64_t kernel_size = 1 + (kernel_size_byte % 7);
        int64_t stride = 1 + (stride_byte % 3);
        int64_t padding = padding_byte % 4;
        int64_t dilation = 1 + (dilation_byte % 3);
        
        // Ensure groups divides in_channels and out_channels
        int64_t max_groups = std::min(in_channels, out_channels);
        int64_t groups = 1 + (groups_byte % max_groups);
        while (in_channels % groups != 0 || out_channels % groups != 0) {
            groups = (groups % max_groups) + 1;
            if (groups > max_groups) groups = 1;
        }
        
        // BatchNorm parameters
        double eps = 1e-5 + (eps_byte / 255.0) * 1e-3;
        double momentum = 0.01 + (momentum_byte / 255.0) * 0.99;
        
        // Padding mode
        std::string padding_mode;
        switch (padding_mode_selector % 4) {
            case 0: padding_mode = "zeros"; break;
            case 1: padding_mode = "reflect"; break;
            case 2: padding_mode = "replicate"; break;
            case 3: padding_mode = "circular"; break;
        }
        
        // Quantization scheme
        torch::QScheme qscheme;
        switch (qscheme_selector % 3) {
            case 0: qscheme = torch::kPerTensorAffine; break;
            case 1: qscheme = torch::kPerChannelAffine; break;
            case 2: qscheme = torch::kPerTensorSymmetric; break;
            default: qscheme = torch::kPerTensorAffine;
        }
        
        // Create ConvBnReLU3d module
        auto conv_bn_relu = torch::nn::intrinsic::qat::ConvBnReLU3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
                .padding_mode(torch::kZeros), // Use enum instead of string for padding_mode
            torch::nn::BatchNorm3dOptions(out_channels)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats),
            torch::nn::ReLUOptions()
        );
        
        // Configure fake quantization if possible
        if (conv_bn_relu->weight_fake_quant) {
            conv_bn_relu->weight_fake_quant->qscheme = qscheme;
            conv_bn_relu->weight_fake_quant->reduce_range = (reduce_range % 2 == 0);
        }
        
        // Create input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure input is 5D for Conv3d (batch, channels, depth, height, width)
            if (input.dim() != 5) {
                // Reshape or create new tensor with correct dimensions
                int64_t batch_size = 1 + (offset < Size ? Data[offset++] % 4 : 1);
                int64_t depth = 1 + (offset < Size ? Data[offset++] % 16 : 8);
                int64_t height = 1 + (offset < Size ? Data[offset++] % 16 : 8);
                int64_t width = 1 + (offset < Size ? Data[offset++] % 16 : 8);
                
                input = torch::randn({batch_size, in_channels, depth, height, width});
            } else {
                // Adjust channel dimension if needed
                if (input.size(1) != in_channels) {
                    auto sizes = input.sizes().vec();
                    sizes[1] = in_channels;
                    input = torch::randn(sizes);
                }
            }
            
            // Ensure input is float for QAT
            if (input.dtype() != torch::kFloat32) {
                input = input.to(torch::kFloat32);
            }
            
        } catch (const std::exception& e) {
            // If tensor creation fails, create a default one
            int64_t batch_size = 2;
            int64_t depth = 8;
            int64_t height = 8;
            int64_t width = 8;
            input = torch::randn({batch_size, in_channels, depth, height, width});
        }
        
        // Set module to training mode for QAT
        conv_bn_relu->train();
        
        // Forward pass
        torch::Tensor output = conv_bn_relu->forward(input);
        
        // Additional operations to increase coverage
        
        // Try eval mode
        conv_bn_relu->eval();
        torch::Tensor eval_output = conv_bn_relu->forward(input);
        
        // Access and modify parameters if they exist
        for (auto& param : conv_bn_relu->parameters()) {
            if (param.defined() && param.requires_grad()) {
                // Trigger gradient computation
                auto sum = param.sum();
                if (sum.requires_grad()) {
                    sum.backward();
                }
            }
        }
        
        // Try to access specific submodules
        auto conv_module = conv_bn_relu->conv;
        auto bn_module = conv_bn_relu->bn;
        auto relu_module = conv_bn_relu->relu;
        
        if (conv_module && bn_module && relu_module) {
            // Test individual components
            auto conv_out = conv_module->forward(input);
            auto bn_out = bn_module->forward(conv_out);
            auto relu_out = relu_module->forward(bn_out);
        }
        
        // Test state_dict functionality
        auto state_dict = conv_bn_relu->state_dict();
        
        // Try to load state dict back
        conv_bn_relu->load_state_dict(state_dict);
        
        // Test different input sizes to stress the module
        if (offset + 3 < Size) {
            int64_t new_depth = 1 + (Data[offset++] % 32);
            int64_t new_height = 1 + (Data[offset++] % 32);
            int64_t new_width = 1 + (Data[offset++] % 32);
            
            torch::Tensor varied_input = torch::randn({1, in_channels, new_depth, new_height, new_width});
            try {
                torch::Tensor varied_output = conv_bn_relu->forward(varied_input);
            } catch (const c10::Error& e) {
                // Expected for some size combinations due to padding/stride
            }
        }
        
        // Test zero-sized batch
        torch::Tensor zero_batch = torch::randn({0, in_channels, 8, 8, 8});
        try {
            torch::Tensor zero_output = conv_bn_relu->forward(zero_batch);
        } catch (const c10::Error& e) {
            // May fail, which is fine
        }
        
        // Test with requires_grad
        input.requires_grad_(true);
        torch::Tensor grad_output = conv_bn_relu->forward(input);
        if (grad_output.requires_grad()) {
            grad_output.sum().backward();
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}