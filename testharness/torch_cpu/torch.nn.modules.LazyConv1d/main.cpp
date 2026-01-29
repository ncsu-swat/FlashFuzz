#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        // Extract parameters for Conv1d
        uint8_t in_channels_byte = Data[offset++];
        uint8_t out_channels_byte = Data[offset++];
        uint8_t kernel_size_byte = Data[offset++];
        uint8_t stride_byte = Data[offset++];
        uint8_t padding_byte = Data[offset++];
        uint8_t dilation_byte = Data[offset++];
        uint8_t groups_byte = Data[offset++];
        uint8_t bias_padding_mode_byte = Data[offset++];
        
        // Convert to reasonable values
        int64_t in_channels = (in_channels_byte % 8) + 1;     // 1-8 input channels
        int64_t out_channels = (out_channels_byte % 16) + 1;  // 1-16 output channels
        int64_t kernel_size = (kernel_size_byte % 7) + 1;     // 1-7 kernel size
        int64_t stride = (stride_byte % 3) + 1;               // 1-3 stride
        int64_t padding = padding_byte % 4;                   // 0-3 padding
        int64_t dilation = (dilation_byte % 2) + 1;           // 1-2 dilation
        int64_t groups = (groups_byte % std::min(in_channels, out_channels)) + 1;
        
        // Ensure in_channels and out_channels are divisible by groups
        in_channels = (in_channels / groups) * groups;
        out_channels = (out_channels / groups) * groups;
        if (in_channels == 0) in_channels = groups;
        if (out_channels == 0) out_channels = groups;
        
        bool bias = (bias_padding_mode_byte & 0x01) == 0;
        uint8_t padding_mode_idx = (bias_padding_mode_byte >> 1) % 4;
        
        // Create input tensor from remaining data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape input tensor to be compatible with Conv1d
        // Conv1d expects input of shape [batch_size, in_channels, sequence_length]
        int64_t total_elements = input_tensor.numel();
        if (total_elements == 0) {
            return 0;
        }
        
        // Calculate minimum sequence length based on kernel and dilation
        int64_t effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1);
        int64_t min_seq_length = effective_kernel;
        
        // Compute sequence length given in_channels
        int64_t seq_length = std::max(min_seq_length, total_elements / in_channels);
        
        // Compute batch_size based on available elements
        int64_t needed_elements = in_channels * seq_length;
        int64_t batch_size = std::max(int64_t(1), total_elements / needed_elements);
        
        // Create properly shaped input
        input_tensor = input_tensor.flatten();
        int64_t actual_elements = batch_size * in_channels * seq_length;
        if (input_tensor.numel() < actual_elements) {
            // Pad with zeros if needed
            torch::Tensor padded = torch::zeros({actual_elements}, input_tensor.options());
            padded.slice(0, 0, input_tensor.numel()).copy_(input_tensor);
            input_tensor = padded;
        } else {
            input_tensor = input_tensor.slice(0, 0, actual_elements);
        }
        input_tensor = input_tensor.reshape({batch_size, in_channels, seq_length});
        
        // Convert to float if not already a floating point type
        if (!torch::isFloatingType(input_tensor.scalar_type())) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Create Conv1d module options
        auto options = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                           .stride(stride)
                           .padding(padding)
                           .dilation(dilation)
                           .groups(groups)
                           .bias(bias);
        
        // Select padding mode
        switch (padding_mode_idx) {
            case 0:
                options.padding_mode(torch::kZeros);
                break;
            case 1:
                options.padding_mode(torch::kReflect);
                break;
            case 2:
                options.padding_mode(torch::kReplicate);
                break;
            case 3:
                options.padding_mode(torch::kCircular);
                break;
        }
        
        torch::nn::Conv1d conv1d(options);
        conv1d->eval();  // Set to eval mode
        
        // Apply the Conv1d operation
        torch::Tensor output;
        try {
            output = conv1d->forward(input_tensor);
        } catch (const c10::Error&) {
            // Shape/size related errors are expected for some inputs
            return 0;
        }
        
        // Perform operations on the output to ensure computation happens
        auto sum = output.sum();
        auto mean = output.mean();
        
        // Access the values to ensure computation is not optimized away
        float sum_val = sum.item<float>();
        float mean_val = mean.item<float>();
        (void)sum_val;
        (void)mean_val;
        
        // Verify module parameters
        auto params = conv1d->parameters();
        for (const auto& p : params) {
            auto p_sum = p.sum().item<float>();
            (void)p_sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}