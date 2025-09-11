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
        
        // Ensure input has at least 3 dimensions (N, C, L) for ConvReLU1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvReLU1d from the remaining data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0;
        int64_t stride = 1, padding = 0, dilation = 1, groups = 1;
        
        if (offset + 3 <= Size) {
            in_channels = Data[offset++] % 8 + 1;  // 1-8 input channels
            out_channels = Data[offset++] % 8 + 1; // 1-8 output channels
            kernel_size = Data[offset++] % 5 + 1;  // 1-5 kernel size
            
            // Ensure input shape is compatible with in_channels
            if (input.size(1) != in_channels) {
                input = input.reshape({input.size(0), in_channels, -1});
            }
            
            // Get additional parameters if available
            if (offset < Size) stride = (Data[offset++] % 3) + 1;     // 1-3
            if (offset < Size) padding = Data[offset++] % 3;          // 0-2
            if (offset < Size) dilation = (Data[offset++] % 2) + 1;   // 1-2
            if (offset < Size) {
                groups = (Data[offset++] % in_channels) + 1;
                // Ensure groups divides in_channels
                if (in_channels % groups != 0) {
                    groups = 1;
                }
            }
        } else {
            // Default values if not enough data
            in_channels = 1;
            out_channels = 1;
            kernel_size = 1;
            
            // Ensure input shape is compatible
            if (input.size(1) != in_channels) {
                input = input.reshape({input.size(0), in_channels, -1});
            }
        }
        
        // Create Conv1d module and apply ReLU manually since ConvReLU1d is not available
        auto conv_options = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                            .stride(stride)
                                            .padding(padding)
                                            .dilation(dilation)
                                            .groups(groups)
                                            .bias(true);
        
        auto conv = torch::nn::Conv1d(conv_options);
        
        // Apply the module to the input tensor and then ReLU
        torch::Tensor conv_output = conv->forward(input);
        torch::Tensor output = torch::relu(conv_output);
        
        // Verify output has expected properties
        if (output.dim() != 3 || 
            output.size(0) != input.size(0) || 
            output.size(1) != out_channels) {
            throw std::runtime_error("Output tensor has unexpected shape");
        }
        
        // Verify ReLU effect (all values should be non-negative)
        if (torch::any(output < 0).item<bool>()) {
            throw std::runtime_error("Output contains negative values, ReLU not applied correctly");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
