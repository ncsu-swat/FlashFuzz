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
        
        // Need at least enough bytes for parameters and tensor data
        if (Size < 8) {
            return 0;
        }
        
        // Parse MaxPool1d parameters first
        int64_t kernel_size = static_cast<int64_t>(Data[offset++]) % 10 + 1; // 1-10
        int64_t stride = static_cast<int64_t>(Data[offset++]) % 10 + 1; // 1-10
        int64_t padding = static_cast<int64_t>(Data[offset++]) % 5; // 0-4
        int64_t dilation = static_cast<int64_t>(Data[offset++]) % 5 + 1; // 1-5
        bool ceil_mode = Data[offset++] % 2 == 1;
        
        // Parse tensor dimensions for 3D input (N, C, L)
        int64_t batch_size = static_cast<int64_t>(Data[offset++]) % 8 + 1; // 1-8
        int64_t channels = static_cast<int64_t>(Data[offset++]) % 16 + 1; // 1-16
        int64_t length = static_cast<int64_t>(Data[offset++]) % 64 + 16; // 16-79 to ensure valid output
        
        // Ensure length is large enough for the pooling operation
        // Effective kernel size considering dilation: dilation * (kernel_size - 1) + 1
        int64_t effective_kernel = dilation * (kernel_size - 1) + 1;
        if (length < effective_kernel) {
            length = effective_kernel + 1;
        }
        
        // Create input tensor with proper 3D shape (N, C, L)
        torch::Tensor input = torch::randn({batch_size, channels, length});
        
        // Use remaining data to influence tensor values
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f; // 0-10
            input = input * scale;
        }
        
        // Create MaxPool1d module
        torch::nn::MaxPool1d pool(torch::nn::MaxPool1dOptions(kernel_size)
                                  .stride(stride)
                                  .padding(padding)
                                  .dilation(dilation)
                                  .ceil_mode(ceil_mode));
        
        // Apply MaxPool1d to the input tensor (inner try-catch for expected failures)
        try {
            torch::Tensor output = pool->forward(input);
            
            // Ensure the output is valid
            if (output.numel() > 0) {
                auto max_val = torch::max(output).item<float>();
                auto sum_val = torch::sum(output).item<float>();
                (void)max_val;
                (void)sum_val;
            }
        } catch (const std::exception &) {
            // Shape mismatches are expected, silently ignore
        }
        
        // Use functional API for max_pool1d_with_indices
        try {
            auto result_tuple = torch::nn::functional::max_pool1d_with_indices(
                input,
                torch::nn::functional::MaxPool1dFuncOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode)
            );
            
            torch::Tensor output_with_indices = std::get<0>(result_tuple);
            torch::Tensor indices = std::get<1>(result_tuple);
            
            // Ensure the outputs are valid
            if (output_with_indices.numel() > 0 && indices.numel() > 0) {
                auto max_val = torch::max(output_with_indices).item<float>();
                auto max_idx = torch::max(indices).item<int64_t>();
                (void)max_val;
                (void)max_idx;
            }
        } catch (const std::exception &) {
            // Shape mismatches are expected, silently ignore
        }
        
        // Also test 2D (unbatched) input: (C, L)
        try {
            torch::Tensor input_2d = torch::randn({channels, length});
            torch::Tensor output_2d = pool->forward(input_2d);
            if (output_2d.numel() > 0) {
                auto sum_val = torch::sum(output_2d).item<float>();
                (void)sum_val;
            }
        } catch (const std::exception &) {
            // Shape mismatches are expected, silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}