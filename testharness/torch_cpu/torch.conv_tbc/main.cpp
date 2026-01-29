#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract dimensions from fuzzer data
        int64_t time_dim = (Data[offset++] % 16) + 1;      // 1-16
        int64_t batch_dim = (Data[offset++] % 8) + 1;      // 1-8
        int64_t in_channels = (Data[offset++] % 16) + 1;   // 1-16
        int64_t out_channels = (Data[offset++] % 16) + 1;  // 1-16
        int64_t kernel_width = (Data[offset++] % 5) + 1;   // 1-5
        int64_t pad = Data[offset++] % 10;                 // 0-9
        
        // Create input tensor in TBC format: [Time, Batch, Channels]
        torch::Tensor input = torch::randn({time_dim, batch_dim, in_channels});
        
        // Create weight tensor: [kernel_width, in_channels, out_channels]
        torch::Tensor weight = torch::randn({kernel_width, in_channels, out_channels});
        
        // Create bias tensor: [out_channels]
        torch::Tensor bias = torch::randn({out_channels});
        
        // Use remaining data to perturb tensor values
        if (offset < Size) {
            torch::Tensor input_noise = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_noise.numel() > 0) {
                int64_t copy_size = std::min(input_noise.numel(), input.numel());
                input.view(-1).slice(0, 0, copy_size).copy_(
                    input_noise.view(-1).slice(0, 0, copy_size));
            }
        }
        
        if (offset < Size) {
            torch::Tensor weight_noise = fuzzer_utils::createTensor(Data, Size, offset);
            if (weight_noise.numel() > 0) {
                int64_t copy_size = std::min(weight_noise.numel(), weight.numel());
                weight.view(-1).slice(0, 0, copy_size).copy_(
                    weight_noise.view(-1).slice(0, 0, copy_size));
            }
        }
        
        if (offset < Size) {
            torch::Tensor bias_noise = fuzzer_utils::createTensor(Data, Size, offset);
            if (bias_noise.numel() > 0) {
                int64_t copy_size = std::min(bias_noise.numel(), bias.numel());
                bias.view(-1).slice(0, 0, copy_size).copy_(
                    bias_noise.view(-1).slice(0, 0, copy_size));
            }
        }
        
        // Inner try-catch for expected operation failures
        try {
            // Apply conv_tbc
            torch::Tensor output = torch::conv_tbc(input, weight, bias, pad);
            
            // Use the output to prevent optimization
            if (output.defined() && output.numel() > 0) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Expected failures (shape mismatches, etc.) - catch silently
        }
        
        // Also test with different dtypes
        if (Size > 32) {
            try {
                torch::Tensor input_f64 = input.to(torch::kFloat64);
                torch::Tensor weight_f64 = weight.to(torch::kFloat64);
                torch::Tensor bias_f64 = bias.to(torch::kFloat64);
                
                torch::Tensor output = torch::conv_tbc(input_f64, weight_f64, bias_f64, pad);
                
                if (output.defined() && output.numel() > 0) {
                    volatile double sum = output.sum().item<double>();
                    (void)sum;
                }
            } catch (const c10::Error&) {
                // Expected failures - catch silently
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}