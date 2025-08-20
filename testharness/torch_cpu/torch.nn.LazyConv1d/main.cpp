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
        
        // Ensure input has at least 3 dimensions for Conv1d (N, C, L)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for Conv1d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1 + (offset < Size ? Data[offset++] % 16 : 1);
        int64_t kernel_size = 1 + (offset < Size ? Data[offset++] % 7 : 1);
        int64_t stride = 1 + (offset < Size ? Data[offset++] % 3 : 1);
        int64_t padding = offset < Size ? Data[offset++] % 3 : 0;
        int64_t dilation = 1 + (offset < Size ? Data[offset++] % 2 : 1);
        int64_t groups = 1;
        
        // Try to set groups to a divisor of in_channels if possible
        if (offset < Size && in_channels > 1) {
            groups = 1 + (Data[offset++] % in_channels);
            if (in_channels % groups != 0) {
                groups = 1; // Fallback to 1 if not a divisor
            }
        }
        
        bool bias = offset < Size ? (Data[offset++] % 2 == 0) : true;
        
        // Create Conv1d module (LazyConv1d is not available in PyTorch C++)
        torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                               .stride(stride)
                               .padding(padding)
                               .dilation(dilation)
                               .groups(groups)
                               .bias(bias));
        
        // Apply the Conv1d operation
        torch::Tensor output = conv->forward(input);
        
        // Force computation to materialize the tensor
        output = output.clone();
        
        // Access some values to ensure computation is performed
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}