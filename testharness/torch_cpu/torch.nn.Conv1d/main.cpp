#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 12) {
            return 0;
        }
        
        // Parse parameters for Conv1d first (before creating input tensor)
        int64_t in_channels = (Data[offset++] % 16) + 1;  // 1-16 channels
        int64_t out_channels = (Data[offset++] % 16) + 1; // 1-16 channels
        int64_t kernel_size = (Data[offset++] % 7) + 1;   // 1-7 kernel size
        int64_t stride = (Data[offset++] % 4) + 1;        // 1-4 stride
        int64_t padding = Data[offset++] % 4;             // 0-3 padding
        int64_t dilation = (Data[offset++] % 3) + 1;      // 1-3 dilation
        bool bias = Data[offset++] % 2 == 0;              // 50% chance of bias
        
        // Groups handling: groups must divide both in_channels and out_channels
        int64_t groups = (Data[offset++] % 4) + 1;  // 1-4 groups
        
        // Adjust in_channels and out_channels to be divisible by groups
        in_channels = groups * ((in_channels + groups - 1) / groups);
        out_channels = groups * ((out_channels + groups - 1) / groups);
        
        // Batch size and sequence length
        int64_t batch_size = (Data[offset++] % 4) + 1;    // 1-4 batch size
        
        // Sequence length must be at least large enough for the effective kernel size
        // effective_kernel_size = dilation * (kernel_size - 1) + 1
        int64_t effective_kernel_size = dilation * (kernel_size - 1) + 1;
        int64_t min_seq_length = effective_kernel_size;
        int64_t seq_length = min_seq_length + (Data[offset++] % 16);  // Add some extra length
        
        // Create input tensor with proper shape [batch_size, in_channels, sequence_length]
        torch::Tensor input = torch::randn({batch_size, in_channels, seq_length});
        
        // Also test with different dtypes based on fuzzer input
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            if (dtype_selector == 1) {
                input = input.to(torch::kDouble);
            }
            // dtype_selector == 0 or 2: keep as float32
        }
        
        // Create Conv1d module with various options
        torch::nn::Conv1dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        // Test different padding modes if we have more data
        if (offset < Size) {
            uint8_t padding_mode = Data[offset++] % 4;
            switch (padding_mode) {
                case 0:
                    options.padding_mode(torch::kZeros);
                    break;
                case 1:
                    options.padding_mode(torch::kReflect);
                    // Reflect padding requires seq_length > padding
                    if (seq_length <= padding) {
                        options.padding(0);
                    }
                    break;
                case 2:
                    options.padding_mode(torch::kReplicate);
                    break;
                case 3:
                    options.padding_mode(torch::kCircular);
                    break;
            }
        }
        
        torch::nn::Conv1d conv(options);
        
        // Match dtype of conv weights with input
        if (input.dtype() == torch::kDouble) {
            conv->to(torch::kDouble);
        }
        
        // Apply Conv1d
        torch::Tensor output = conv->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        auto max_val = output.max();
        
        // Test backward pass if possible
        if (output.requires_grad() || input.requires_grad()) {
            try {
                output.sum().backward();
            } catch (...) {
                // Backward might fail in some cases, that's ok
            }
        }
        
        // Ensure the operations are not optimized away
        if (sum.item<double>() == -1.0 && mean.item<double>() == -1.0 && max_val.item<double>() == -1.0) {
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