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
        
        // Early return if not enough data
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
            
            if (offset < Size) {
                bias = Data[offset] % 2 == 0;
                offset++;
            }
        }
        
        // Reshape input to match expected dimensions for ConvTranspose1d
        if (input.size(1) != in_channels) {
            input = input.reshape({input.size(0), in_channels, -1});
        }
        
        // Create weight tensor for the convolution
        std::vector<int64_t> weight_shape = {in_channels, out_channels / groups, kernel_size};
        auto weight_options = torch::TensorOptions().dtype(torch::kFloat);
        auto weight = torch::rand(weight_shape, weight_options);
        
        // Create bias tensor if needed
        torch::Tensor bias_tensor;
        if (bias) {
            bias_tensor = torch::rand({out_channels}, weight_options);
        }
        
        // Create ConvTranspose1d module options
        torch::nn::ConvTranspose1dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride);
        options.padding(padding);
        options.output_padding(output_padding);
        options.dilation(dilation);
        options.groups(groups);
        options.bias(bias);
            
        auto conv_transpose = torch::nn::ConvTranspose1d(options);
        
        // Set the weights and bias
        conv_transpose->weight = weight;
        if (bias) {
            conv_transpose->bias = bias_tensor;
        }
        
        // Forward pass
        auto output = conv_transpose->forward(input);
        
        // Try different input types
        if (offset + 1 < Size) {
            auto dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try with a different data type if possible
            try {
                auto input2 = input.to(dtype);
                auto output2 = conv_transpose->forward(input2);
            } catch (const std::exception& e) {
                // Ignore exceptions from data type conversion
            }
        }
        
        // Try with a different shape if possible
        if (offset + 4 < Size) {
            try {
                auto input3 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                if (input3.dim() < 3) {
                    input3 = input3.reshape({1, in_channels, -1});
                } else if (input3.size(1) != in_channels) {
                    input3 = input3.reshape({input3.size(0), in_channels, -1});
                }
                auto output3 = conv_transpose->forward(input3);
            } catch (const std::exception& e) {
                // Ignore exceptions from shape conversion
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
