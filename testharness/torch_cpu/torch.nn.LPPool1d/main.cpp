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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 3 more bytes for parameters
        if (Size - offset < 3) {
            return 0;
        }
        
        // Parse parameters for LPPool1d
        uint8_t norm_type_byte = Data[offset++];
        int64_t norm_type = (norm_type_byte % 10) + 1; // Norm type between 1 and 10
        
        uint8_t kernel_size_byte = Data[offset++];
        int64_t kernel_size = (kernel_size_byte % 8) + 1; // Kernel size between 1 and 8
        
        uint8_t stride_byte = Data[offset++];
        int64_t stride = (stride_byte % 4) + 1; // Stride between 1 and 4
        
        // Create LPPool1d module
        torch::nn::LPPool1d lppool(
            torch::nn::LPPool1dOptions(norm_type, kernel_size)
                .stride(stride)
        );
        
        // Apply LPPool1d to the input tensor
        torch::Tensor output;
        
        // Ensure input has at least 2 dimensions (batch_size, channels, length)
        if (input.dim() < 2) {
            // Add dimensions if needed
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                input = input.unsqueeze(0);
            }
        }
        
        // Add a third dimension if needed (for 1D pooling)
        if (input.dim() == 2) {
            input = input.unsqueeze(-1);
        }
        
        // Apply the operation
        output = lppool->forward(input);
        
        // Verify output is not empty
        if (output.numel() > 0) {
            // Access some elements to ensure computation happened
            auto sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
