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
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 1 byte left for parameters
        if (offset >= Size) {
            return 0;
        }
        
        // Parse parameters for Conv1d
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Extract parameters from remaining data
        if (offset < Size) {
            in_channels = (Data[offset++] % 16) + 1; // 1-16 channels
        }
        
        if (offset < Size) {
            out_channels = (Data[offset++] % 16) + 1; // 1-16 channels
        }
        
        if (offset < Size) {
            kernel_size = (Data[offset++] % 7) + 1; // 1-7 kernel size
        }
        
        if (offset < Size) {
            stride = (Data[offset++] % 4) + 1; // 1-4 stride
        }
        
        if (offset < Size) {
            padding = Data[offset++] % 4; // 0-3 padding
        }
        
        if (offset < Size) {
            dilation = (Data[offset++] % 3) + 1; // 1-3 dilation
        }
        
        if (offset < Size) {
            // Ensure groups divides in_channels
            groups = (Data[offset++] % in_channels) + 1;
            if (groups > 1) {
                // Ensure in_channels is divisible by groups
                in_channels = groups * ((in_channels / groups) + 1);
            }
        }
        
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0; // 50% chance of bias
        }
        
        // Reshape input tensor if needed to match Conv1d requirements
        // Conv1d expects input of shape [batch_size, in_channels, sequence_length]
        if (input.dim() < 3) {
            // Create a new shape with at least 3 dimensions
            std::vector<int64_t> new_shape;
            
            // Add batch dimension if needed
            if (input.dim() == 0) {
                new_shape.push_back(1); // batch size
                new_shape.push_back(in_channels); // channels
                new_shape.push_back(8); // sequence length
            } else if (input.dim() == 1) {
                new_shape.push_back(1); // batch size
                new_shape.push_back(in_channels); // channels
                new_shape.push_back(input.size(0)); // use existing dim as sequence length
            } else if (input.dim() == 2) {
                new_shape.push_back(input.size(0)); // use first dim as batch
                new_shape.push_back(in_channels); // channels
                new_shape.push_back(input.size(1)); // use second dim as sequence length
            }
            
            // Reshape or create new tensor
            input = torch::ones(new_shape, input.options());
        } else {
            // If tensor already has 3+ dimensions, ensure channel dim matches in_channels
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[1] = in_channels; // Set channel dimension
            input = torch::ones(new_shape, input.options());
        }
        
        // Create Conv1d module
        torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                .stride(stride)
                                .padding(padding)
                                .dilation(dilation)
                                .groups(groups)
                                .bias(bias));
        
        // Apply Conv1d
        torch::Tensor output = conv->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        auto max_val = output.max();
        
        // Ensure the operations are not optimized away
        if (sum.item<float>() == -1.0f && mean.item<float>() == -1.0f && max_val.item<float>() == -1.0f) {
            return 1; // This condition is unlikely to be true
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
