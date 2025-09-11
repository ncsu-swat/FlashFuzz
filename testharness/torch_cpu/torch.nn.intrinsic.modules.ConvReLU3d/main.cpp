#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (batch, channels, depth, height, width)
        if (input.dim() != 5) {
            input = input.reshape({1, 1, 1, 1, 1});
        }
        
        // Extract parameters for Conv3d from the remaining data
        uint8_t in_channels = 1;
        uint8_t out_channels = 1;
        uint8_t kernel_size = 1;
        uint8_t stride = 1;
        uint8_t padding = 0;
        uint8_t dilation = 1;
        bool bias = true;
        
        if (offset + 7 <= Size) {
            in_channels = std::max(1, static_cast<int>(Data[offset++]));
            out_channels = std::max(1, static_cast<int>(Data[offset++]));
            kernel_size = std::max(1, static_cast<int>(Data[offset++]) % 5 + 1);
            stride = std::max(1, static_cast<int>(Data[offset++]) % 3 + 1);
            padding = static_cast<int>(Data[offset++]) % 3;
            dilation = std::max(1, static_cast<int>(Data[offset++]) % 2 + 1);
            bias = Data[offset++] % 2 == 0;
        }
        
        // Reshape input to match in_channels if needed
        if (input.size(1) != in_channels) {
            std::vector<int64_t> new_shape = {input.size(0), in_channels};
            for (int i = 2; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            input = input.reshape(new_shape);
        }
        
        // Create Conv3d module and apply ReLU manually
        torch::nn::Conv3dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .bias(bias);
        
        auto conv3d = torch::nn::Conv3d(options);
        
        // Forward pass with ReLU activation
        torch::Tensor output = torch::relu(conv3d->forward(input));
        
        // Test with different input types
        if (offset + 1 <= Size) {
            auto dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try with different dtype if supported for convolution
            if (dtype == torch::kFloat || dtype == torch::kDouble) {
                input = input.to(dtype);
                conv3d = torch::nn::Conv3d(options);
                output = torch::relu(conv3d->forward(input));
            }
        }
        
        // Test with different input shapes
        if (offset + 3 <= Size) {
            int64_t depth = std::max(1, static_cast<int>(Data[offset++]) % 10 + 1);
            int64_t height = std::max(1, static_cast<int>(Data[offset++]) % 10 + 1);
            int64_t width = std::max(1, static_cast<int>(Data[offset++]) % 10 + 1);
            
            input = torch::ones({1, in_channels, depth, height, width});
            output = torch::relu(conv3d->forward(input));
        }
        
        // Test with zero-sized dimensions (should trigger exceptions handled by try-catch)
        if (offset + 1 <= Size && Data[offset++] % 5 == 0) {
            try {
                input = torch::ones({1, in_channels, 0, 1, 1});
                output = torch::relu(conv3d->forward(input));
            } catch (...) {
                // Expected exception for invalid input
            }
        }
        
        // Test with very large dimensions (may cause memory issues)
        if (offset + 1 <= Size && Data[offset++] % 20 == 0) {
            try {
                int64_t large_dim = std::min(static_cast<int64_t>(100), 
                                            static_cast<int64_t>(Data[offset++]) + 50);
                input = torch::ones({1, in_channels, large_dim, large_dim, large_dim});
                output = torch::relu(conv3d->forward(input));
            } catch (...) {
                // Expected exception for memory issues
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
