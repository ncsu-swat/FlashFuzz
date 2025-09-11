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
        
        // Ensure we have enough data left for parameters
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Extract parameters for the conv module
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 0;
        int64_t padding = 0;
        int64_t dilation = 0;
        int64_t groups = 0;
        bool bias = false;
        
        // Parse parameters from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&in_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            in_channels = std::abs(in_channels) % 16 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 16 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_size = std::abs(kernel_size) % 7 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride = std::abs(stride) % 3 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding = std::abs(padding) % 3; // Can be 0
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation = std::abs(dilation) % 3 + 1; // Ensure positive and reasonable
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % in_channels + 1; // Ensure positive and reasonable
            
            // Ensure groups divides in_channels
            if (in_channels % groups != 0) {
                groups = 1;
            }
        } else {
            groups = 1;
        }
        
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0; // 50% chance of having bias
        }
        
        // Create Conv module (QAT modules are not directly available in C++ API)
        torch::nn::Conv2d conv = nullptr;
        
        // Determine the type of conv module based on input dimensions
        if (input.dim() == 4) {
            // Conv2d for 4D input (batch_size, channels, height, width)
            conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .groups(groups)
                    .bias(bias)
            );
        } else if (input.dim() == 3) {
            // Try to reshape to 4D by adding batch dimension
            input = input.unsqueeze(0);
            conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .groups(groups)
                    .bias(bias)
            );
        } else {
            // For other dimensions, create a default Conv2d and reshape input
            conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .groups(groups)
                    .bias(bias)
            );
            
            // Reshape input to match expected dimensions for Conv2d
            std::vector<int64_t> new_shape = {1, in_channels, 8, 8}; // Default shape
            input = input.reshape(new_shape);
        }
        
        // Apply the Conv module
        torch::Tensor output;
        try {
            output = conv->forward(input);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and handled
            return 0;
        }
        
        // Try to access and use the output to prevent optimization
        if (output.defined()) {
            auto sum = output.sum().item<float>();
            if (std::isnan(sum) || std::isinf(sum)) {
                return 0;
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
