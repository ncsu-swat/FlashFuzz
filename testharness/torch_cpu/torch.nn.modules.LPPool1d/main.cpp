#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for LPPool1d first
        uint8_t norm_type_byte = Data[offset++];
        uint8_t kernel_size_byte = Data[offset++];
        uint8_t stride_byte = Data[offset++];
        uint8_t batch_byte = Data[offset++];
        uint8_t channels_byte = Data[offset++];
        uint8_t length_byte = Data[offset++];
        
        // Normalize parameters
        double norm_type = static_cast<double>(norm_type_byte % 10) + 1.0; // Norm type between 1 and 10
        int64_t kernel_size = static_cast<int64_t>(kernel_size_byte % 7) + 1; // Kernel size between 1 and 7
        int64_t stride = static_cast<int64_t>(stride_byte % 5) + 1; // Stride between 1 and 5
        
        // Create proper 3D tensor dimensions for LPPool1d: (N, C, L)
        int64_t batch = static_cast<int64_t>(batch_byte % 4) + 1;      // Batch 1-4
        int64_t channels = static_cast<int64_t>(channels_byte % 8) + 1; // Channels 1-8
        int64_t length = static_cast<int64_t>(length_byte % 32) + kernel_size; // Length must be >= kernel_size
        
        // Create input tensor with proper shape for LPPool1d
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            // Reshape to 3D (N, C, L) for LPPool1d
            input = input.reshape({batch, channels, length}).to(torch::kFloat32);
        } catch (...) {
            // If reshape fails, create a tensor directly
            input = torch::randn({batch, channels, length}, torch::kFloat32);
        }
        
        // Ensure input requires float type and is contiguous
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        input = input.contiguous();
        
        // Create LPPool1d module
        torch::nn::LPPool1d lppool(
            torch::nn::LPPool1dOptions(norm_type, kernel_size)
                .stride(stride)
        );
        
        // Apply LPPool1d to the input tensor
        torch::Tensor output;
        try {
            output = lppool->forward(input);
        } catch (const c10::Error &e) {
            // Expected failures due to shape mismatches, etc.
            return 0;
        }
        
        // Ensure the output is valid by performing a simple operation
        try {
            auto sum = output.sum();
            
            // Check if the sum is finite (use double for more precision)
            if (output.numel() > 0) {
                volatile float val = sum.item<float>();
                (void)val;
            }
        } catch (...) {
            // Silent catch for validation operations
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}