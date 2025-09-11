#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvTranspose1d
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        if (offset + 8 <= Size) {
            in_channels = (Data[offset] % 8) + 1;
            offset++;
            out_channels = (Data[offset] % 8) + 1;
            offset++;
            kernel_size = (Data[offset] % 5) + 1;
            offset++;
            stride = (Data[offset] % 3) + 1;
            offset++;
            padding = Data[offset] % 3;
            offset++;
            output_padding = Data[offset] % std::min(stride - 1, static_cast<int64_t>(1));
            offset++;
            dilation = (Data[offset] % 2) + 1;
            offset++;
            groups = std::gcd(in_channels, out_channels);
            if (groups > 1) {
                groups = (Data[offset] % groups) + 1;
            }
            offset++;
            bias = (offset < Size) ? (Data[offset] % 2 == 0) : true;
            offset++;
        } else {
            in_channels = 1;
            out_channels = 1;
            kernel_size = 1;
        }
        
        // Reshape input to match in_channels if needed
        if (input.size(1) != in_channels) {
            auto shape = input.sizes().vec();
            shape[1] = in_channels;
            input = input.reshape(shape);
        }
        
        // Create ConvTranspose1d module
        torch::nn::ConvTranspose1dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .output_padding(output_padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv_transpose = torch::nn::ConvTranspose1d(options);
        
        // Apply the operation
        auto output = conv_transpose(input);
        
        // Try with different quantization parameters if there's more data
        if (offset + 2 <= Size) {
            double scale = (Data[offset] % 100) / 100.0 + 0.01;
            offset++;
            int64_t zero_point = Data[offset] % 256 - 128;
            offset++;
            
            // Quantize the input tensor
            auto quantized_input = torch::quantize_per_tensor(
                input.to(torch::kFloat), 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
            
            // Try to run with quantized input
            try {
                auto dequantized = torch::dequantize(quantized_input);
                auto output2 = conv_transpose(dequantized);
            } catch (...) {
                // Ignore errors from quantization
            }
        }
        
        // Try with different weight formats if there's more data
        if (offset + 1 <= Size) {
            try {
                // Get the weight and bias
                auto params = conv_transpose->parameters();
                
                // Try to pack the weight
                auto packed_weight = torch::_empty_affine_quantized(
                    {out_channels, in_channels / groups, kernel_size},
                    torch::kQUInt8,
                    0.1,  // scale
                    0     // zero_point
                );
                
                // Try to set the packed weight
                conv_transpose->weight = packed_weight;
                
                // Try to run with packed weight
                auto output3 = conv_transpose(input);
            } catch (...) {
                // Ignore errors from weight packing
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
