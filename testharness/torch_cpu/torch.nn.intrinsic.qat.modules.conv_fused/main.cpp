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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for the conv module
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Extract parameters from the remaining data
        if (offset + 8 <= Size) {
            in_channels = 1 + (Data[offset++] % 16);
            out_channels = 1 + (Data[offset++] % 16);
            kernel_size = 1 + (Data[offset++] % 7);
            
            if (offset < Size) stride = 1 + (Data[offset++] % 3);
            if (offset < Size) padding = Data[offset++] % 3;
            if (offset < Size) dilation = 1 + (Data[offset++] % 2);
            if (offset < Size) groups = 1 + (Data[offset++] % std::min(in_channels, out_channels));
            if (offset < Size) bias = Data[offset++] % 2 == 0;
        } else {
            in_channels = 3;
            out_channels = 6;
            kernel_size = 3;
        }
        
        // Ensure groups divides both in_channels and out_channels
        if (in_channels % groups != 0 || out_channels % groups != 0) {
            groups = 1;
        }
        
        // Reshape input tensor if needed to match conv requirements
        // For Conv1d: [batch_size, in_channels, width]
        // For Conv2d: [batch_size, in_channels, height, width]
        // For Conv3d: [batch_size, in_channels, depth, height, width]
        
        int ndim = input.dim();
        torch::Tensor reshaped_input;
        
        if (ndim == 0) {
            // Scalar to 3D tensor [1, in_channels, 1]
            reshaped_input = torch::ones({1, in_channels, 1}, input.options());
            ndim = 3;
        } else if (ndim == 1) {
            // 1D to 3D tensor [1, in_channels, length]
            int64_t length = input.size(0);
            reshaped_input = input.reshape({1, in_channels, std::max(int64_t(1), length / in_channels)});
            ndim = 3;
        } else if (ndim == 2) {
            // 2D to 3D tensor [batch, in_channels, width]
            reshaped_input = input.reshape({input.size(0), in_channels, std::max(int64_t(1), input.size(1) / in_channels)});
            ndim = 3;
        } else if (ndim >= 3) {
            // Ensure the second dimension is in_channels
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[1] = in_channels;
            reshaped_input = torch::ones(new_shape, input.options());
            ndim = reshaped_input.dim();
        }
        
        // Create the appropriate conv module based on input dimensions
        torch::nn::Module* conv_module = nullptr;
        
        if (ndim == 3) {
            // Conv1d
            auto conv = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                         .stride(stride)
                                         .padding(padding)
                                         .dilation(dilation)
                                         .groups(groups)
                                         .bias(bias));
            
            auto bn = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(out_channels));
            auto relu = torch::nn::ReLU();
            
            // Create sequential module to simulate fused behavior
            auto sequential = torch::nn::Sequential(conv, bn, relu);
            
            // Apply the module
            auto output = sequential->forward(reshaped_input);
        } else if (ndim == 4) {
            // Conv2d
            auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                         .stride(stride)
                                         .padding(padding)
                                         .dilation(dilation)
                                         .groups(groups)
                                         .bias(bias));
            
            auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
            auto relu = torch::nn::ReLU();
            
            // Create sequential module to simulate fused behavior
            auto sequential = torch::nn::Sequential(conv, bn, relu);
            
            // Apply the module
            auto output = sequential->forward(reshaped_input);
        } else if (ndim == 5) {
            // Conv3d
            auto conv = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                                         .stride(stride)
                                         .padding(padding)
                                         .dilation(dilation)
                                         .groups(groups)
                                         .bias(bias));
            
            auto bn = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(out_channels));
            auto relu = torch::nn::ReLU();
            
            // Create sequential module to simulate fused behavior
            auto sequential = torch::nn::Sequential(conv, bn, relu);
            
            // Apply the module
            auto output = sequential->forward(reshaped_input);
        }
        
        // Test quantization-aware training (QAT) functionality
        if (ndim == 4) {  // For Conv2d which is most common
            // Create a QAT model with fused modules
            torch::nn::Conv2d conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                     .stride(stride)
                                                     .padding(padding)
                                                     .dilation(dilation)
                                                     .groups(groups)
                                                     .bias(bias));
            
            // Simulate the QAT process
            // In real QAT, we would use torch::nn::intrinsic::qat::ConvBnReLU2d
            // but we'll simulate with standard modules for testing
            
            // Create fake quantization parameters
            double scale = 1.0 / 128.0;
            int zero_point = 128;
            
            // Apply fake quantization to input
            auto q_input = torch::fake_quantize_per_tensor_affine(
                reshaped_input, scale, zero_point, 0, 255);
            
            // Forward pass through conv
            auto conv_out = conv->forward(q_input);
            
            // Apply fake quantization to weights
            auto weight = conv->weight;
            auto q_weight = torch::fake_quantize_per_tensor_affine(
                weight, scale, zero_point, -128, 127);
            
            // Apply fake quantization to output
            auto q_output = torch::fake_quantize_per_tensor_affine(
                conv_out, scale, zero_point, 0, 255);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}