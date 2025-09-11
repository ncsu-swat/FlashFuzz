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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions for Conv1d (batch_size, channels, length)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for Conv1d from the remaining data
        uint8_t in_channels = 0, out_channels = 0, kernel_size = 0;
        int64_t stride = 1, padding = 0, dilation = 1, groups = 1;
        bool bias = true;
        
        if (offset + 3 <= Size) {
            in_channels = Data[offset++] % 16 + 1;  // 1-16 channels
            out_channels = Data[offset++] % 16 + 1; // 1-16 channels
            kernel_size = Data[offset++] % 7 + 1;   // 1-7 kernel size
        }
        
        if (offset + 4 <= Size) {
            stride = (Data[offset++] % 3) + 1;      // 1-3 stride
            padding = Data[offset++] % 3;           // 0-2 padding
            dilation = (Data[offset++] % 2) + 1;    // 1-2 dilation
            groups = (Data[offset++] % in_channels) + 1; // 1-in_channels groups
            
            // Ensure groups divides in_channels
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        if (offset < Size) {
            bias = Data[offset++] & 1; // 0 or 1 for bias
        }
        
        // Ensure input shape is compatible with Conv1d
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t length = input.size(2);
        
        // Reshape input to match in_channels if needed
        if (channels != in_channels) {
            input = input.reshape({batch_size, in_channels, -1});
            length = input.size(2);
        }
        
        // Create regular Conv1d module (quantized dynamic modules are not available in C++ frontend)
        torch::nn::Conv1dOptions options = 
            torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias);
        
        auto conv1d = torch::nn::Conv1d(options);
        
        // Apply the Conv1d operation
        torch::Tensor output = conv1d->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        auto max_val = output.max();
        
        // Prevent compiler from optimizing away the operations
        if (sum.item<float>() == -1.0f && mean.item<float>() == -1.0f && max_val.item<float>() == -1.0f) {
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
