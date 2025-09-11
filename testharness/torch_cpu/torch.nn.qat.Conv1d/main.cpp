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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (batch_size, in_channels, sequence_length)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for Conv1d from the remaining data
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Parse parameters if we have enough data
        if (offset + 8 <= Size) {
            out_channels = (Data[offset] % 8) + 1;
            offset++;
            
            kernel_size = (Data[offset] % 5) + 1;
            offset++;
            
            stride = (Data[offset] % 3) + 1;
            offset++;
            
            padding = Data[offset] % 3;
            offset++;
            
            dilation = (Data[offset] % 2) + 1;
            offset++;
            
            // Ensure groups divides in_channels
            groups = (Data[offset] % in_channels) + 1;
            offset++;
            
            bias = Data[offset] % 2 == 0;
            offset++;
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset < Size) {
            // Use remaining data to influence scale
            scale = 0.1 + (static_cast<double>(Data[offset]) / 255.0);
            offset++;
        }
        
        if (offset < Size) {
            // Use remaining data to influence zero_point
            zero_point = static_cast<int64_t>(Data[offset]) % 256;
            offset++;
        }
        
        // Create regular Conv1d module (QAT modules are not available in C++ frontend)
        torch::nn::Conv1d conv(
            torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(groups)
                .bias(bias)
        );
        
        // Forward pass
        torch::Tensor output = conv->forward(input);
        
        // Try to access some properties of the output to ensure computation happened
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try to perform some operations on the output
        if (output.numel() > 0) {
            auto sum = output.sum();
            auto mean = output.mean();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
