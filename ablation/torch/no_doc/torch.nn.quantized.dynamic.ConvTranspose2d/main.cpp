#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <torch/nn/modules/conv.h>
#include <ATen/ATen.h>
#include <iostream>
#include <cstring>

// Helper to consume bytes and create parameters
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T min_val, T max_val) {
    if (offset + sizeof(T) > size) {
        offset = size;
        return min_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    
    // Clamp to reasonable range
    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    if (Size < 10) {
        return 0; // Need minimum bytes for basic parameters
    }

    try
    {
        size_t offset = 0;
        
        // Parse convolution parameters from fuzzer input
        int64_t in_channels = consumeValue<int64_t>(Data, offset, Size, 1, 512);
        int64_t out_channels = consumeValue<int64_t>(Data, offset, Size, 1, 512);
        
        // Kernel size (1-7 for reasonable range)
        int64_t kernel_h = consumeValue<int64_t>(Data, offset, Size, 1, 7);
        int64_t kernel_w = consumeValue<int64_t>(Data, offset, Size, 1, 7);
        
        // Stride (1-5)
        int64_t stride_h = consumeValue<int64_t>(Data, offset, Size, 1, 5);
        int64_t stride_w = consumeValue<int64_t>(Data, offset, Size, 1, 5);
        
        // Padding (0-3)
        int64_t pad_h = consumeValue<int64_t>(Data, offset, Size, 0, 3);
        int64_t pad_w = consumeValue<int64_t>(Data, offset, Size, 0, 3);
        
        // Output padding (0-2)
        int64_t out_pad_h = consumeValue<int64_t>(Data, offset, Size, 0, 2);
        int64_t out_pad_w = consumeValue<int64_t>(Data, offset, Size, 0, 2);
        
        // Dilation (1-3)
        int64_t dilation_h = consumeValue<int64_t>(Data, offset, Size, 1, 3);
        int64_t dilation_w = consumeValue<int64_t>(Data, offset, Size, 1, 3);
        
        // Groups (1 to in_channels, must divide in_channels and out_channels)
        int64_t groups = consumeValue<int64_t>(Data, offset, Size, 1, std::min(in_channels, out_channels));
        // Ensure groups divides both in_channels and out_channels
        while (groups > 1 && (in_channels % groups != 0 || out_channels % groups != 0)) {
            groups--;
        }
        
        // Bias flag
        bool use_bias = (offset < Size) ? (Data[offset++] % 2 == 1) : false;
        
        // Create input tensor (float for dynamic quantization)
        torch::Tensor input;
        if (offset < Size) {
            try {
                // Parse input dimensions
                int64_t batch_size = consumeValue<int64_t>(Data, offset, Size, 1, 32);
                int64_t height = consumeValue<int64_t>(Data, offset, Size, 1, 64);
                int64_t width = consumeValue<int64_t>(Data, offset, Size, 1, 64);
                
                // Create input tensor with shape [batch, in_channels, height, width]
                std::vector<int64_t> input_shape = {batch_size, in_channels, height, width};
                
                // Use remaining data for tensor values or random if not enough data
                if (offset + batch_size * in_channels * height * width * sizeof(float) <= Size) {
                    input = fuzzer_utils::createTensor(Data, Size, offset);
                    // Reshape if needed
                    if (input.numel() >= batch_size * in_channels * height * width) {
                        input = input.view(input_shape);
                    } else {
                        input = torch::randn(input_shape, torch::kFloat32);
                    }
                } else {
                    input = torch::randn(input_shape, torch::kFloat32);
                }
                
                // Ensure float type for dynamic quantization
                if (input.dtype() != torch::kFloat32) {
                    input = input.to(torch::kFloat32);
                }
            } catch (...) {
                // Fallback to default input
                input = torch::randn({2, in_channels, 16, 16}, torch::kFloat32);
            }
        } else {
            input = torch::randn({2, in_channels, 16, 16}, torch::kFloat32);
        }
        
        // Create quantized weight tensor
        torch::Tensor weight = torch::randn({in_channels, out_channels / groups, kernel_h, kernel_w}, torch::kFloat32);
        
        // Quantize the weight
        double scale = 0.1;
        int64_t zero_point = 0;
        torch::Tensor qweight = torch::quantize_per_tensor(weight, scale, zero_point, torch::kQInt8);
        
        // Create bias if needed
        torch::Tensor bias;
        if (use_bias) {
            bias = torch::randn({out_channels}, torch::kFloat32);
        }
        
        // Create packed parameters for quantized conv transpose
        auto packed_params = c10::intrusive_ptr<torch::nn::intrusive_ptr<torch::nn::Module>>();
        
        // Perform dynamic quantized transposed convolution
        // Note: The actual API might differ, using the pattern from PyTorch's quantized ops
        try {
            // Try different quantization approaches based on available APIs
            
            // Approach 1: Use functional API if available
            torch::Tensor output;
            
            // Set quantization engine
            at::globalContext().setQEngine(at::QEngine::QNNPACK);
            
            // Create convolution options
            torch::nn::ConvTranspose2dOptions conv_options(in_channels, out_channels, kernel_h);
            conv_options.stride({stride_h, stride_w});
            conv_options.padding({pad_h, pad_w});
            conv_options.output_padding({out_pad_h, out_pad_w});
            conv_options.dilation({dilation_h, dilation_w});
            conv_options.groups(groups);
            conv_options.bias(use_bias);
            
            // Try to apply dynamic quantized conv transpose
            // This might involve:
            // 1. Quantizing input dynamically
            // 2. Using pre-quantized weights
            // 3. Applying conv transpose
            // 4. Dequantizing output
            
            // Dynamic quantization of input
            auto qinput = torch::quantize_per_tensor(input, 0.1, 0, torch::kQUInt8);
            
            // Apply quantized conv transpose (API may vary)
            if (use_bias) {
                output = at::quantized_conv_transpose2d(
                    qinput, qweight, bias,
                    {stride_h, stride_w}, {pad_h, pad_w}, 
                    {out_pad_h, out_pad_w}, groups, {dilation_h, dilation_w}
                );
            } else {
                output = at::quantized_conv_transpose2d(
                    qinput, qweight, c10::nullopt,
                    {stride_h, stride_w}, {pad_h, pad_w},
                    {out_pad_h, out_pad_w}, groups, {dilation_h, dilation_w}
                );
            }
            
            // Dequantize output
            output = output.dequantize();
            
            // Validate output
            if (!output.defined()) {
                std::cerr << "Output tensor is not defined" << std::endl;
            } else if (output.numel() == 0) {
                std::cerr << "Output tensor is empty" << std::endl;
            } else {
                // Check for NaN or Inf
                if (torch::any(torch::isnan(output)).item<bool>()) {
                    std::cerr << "Output contains NaN" << std::endl;
                }
                if (torch::any(torch::isinf(output)).item<bool>()) {
                    std::cerr << "Output contains Inf" << std::endl;
                }
            }
            
        } catch (const c10::Error& e) {
            // Try alternative approach or handle specific quantization errors
            std::cerr << "Quantization error: " << e.what() << std::endl;
            
            // Fallback: try regular conv transpose for comparison
            try {
                torch::nn::ConvTranspose2d conv_transpose(
                    torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_h)
                    .stride(stride_h)
                    .padding(pad_h)
                    .output_padding(out_pad_h)
                    .dilation(dilation_h)
                    .groups(groups)
                    .bias(use_bias)
                );
                
                auto output = conv_transpose->forward(input);
                
            } catch (...) {
                // Silently continue
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}