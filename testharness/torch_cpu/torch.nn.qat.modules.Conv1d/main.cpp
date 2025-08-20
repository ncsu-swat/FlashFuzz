#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for Conv1d
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
            padding = Data[offset++] % 4;           // 0-3 padding
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
        
        // Create regular Conv1d module (QAT modules are not available in C++ frontend)
        torch::nn::Conv1dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv1d = torch::nn::Conv1d(options);
        
        // Ensure input channels match the module's expected input
        if (input.size(1) != in_channels) {
            input = input.reshape({input.size(0), in_channels, -1});
        }
        
        // Apply the operation
        torch::Tensor output = conv1d->forward(input);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}