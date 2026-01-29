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
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions for Conv1d (batch_size, channels, length)
        if (input.dim() < 3) {
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                new_shape = {1, 1, 1};
            } else if (input.dim() == 1) {
                int64_t len = input.size(0);
                if (len == 0) len = 1;
                new_shape = {1, 1, len};
            } else if (input.dim() == 2) {
                int64_t batch = input.size(0);
                int64_t len = input.size(1);
                if (batch == 0) batch = 1;
                if (len == 0) len = 1;
                new_shape = {batch, 1, len};
            }
            
            int64_t total_elements = input.numel();
            int64_t new_total = 1;
            for (auto s : new_shape) new_total *= s;
            
            if (total_elements == 0 || new_total == 0) {
                input = torch::zeros(new_shape);
            } else if (total_elements != new_total) {
                input = torch::zeros(new_shape);
            } else {
                input = input.reshape(new_shape);
            }
        }
        
        // Ensure input tensor is float for convolution
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Extract parameters for Conv1d from the remaining data
        int64_t in_channels = input.size(1);
        if (in_channels <= 0) in_channels = 1;
        
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
            
            groups = (Data[offset] % std::max(in_channels, int64_t(1))) + 1;
            while (groups > 1 && (in_channels % groups != 0 || out_channels % groups != 0)) {
                groups--;
            }
            offset++;
            
            bias = (Data[offset] % 2) == 0;
            offset++;
        }
        
        // Ensure input length is sufficient for convolution
        int64_t effective_kernel = (kernel_size - 1) * dilation + 1;
        int64_t min_length = effective_kernel;
        if (input.size(2) < min_length) {
            input = torch::nn::functional::pad(
                input, 
                torch::nn::functional::PadFuncOptions({0, min_length - input.size(2)})
            );
        }
        
        // Create Conv1d module
        torch::nn::Conv1d conv(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                .stride(stride)
                                .padding(padding)
                                .dilation(dilation)
                                .groups(groups)
                                .bias(bias));
        
        // Apply Conv1d
        torch::Tensor output = conv->forward(input);
        
        // Try different input types
        if (offset + 1 < Size) {
            try {
                torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                int64_t numel = input2.numel();
                if (numel > 0) {
                    int64_t batch = 1;
                    int64_t length = numel / in_channels;
                    if (length == 0) length = 1;
                    
                    input2 = torch::zeros({batch, in_channels, length});
                    
                    if (!input2.is_floating_point()) {
                        input2 = input2.to(torch::kFloat);
                    }
                    
                    if (input2.size(2) < min_length) {
                        input2 = torch::nn::functional::pad(
                            input2,
                            torch::nn::functional::PadFuncOptions({0, min_length - input2.size(2)})
                        );
                    }
                    
                    torch::Tensor output2 = conv->forward(input2);
                }
            } catch (...) {
                // Silently ignore shape-related errors for secondary inputs
            }
        }
        
        // Try with different padding modes if we have more data
        if (offset + 1 < Size) {
            uint8_t padding_selector = Data[offset++] % 4;
            
            try {
                torch::nn::Conv1dOptions opts(in_channels, out_channels, kernel_size);
                opts.stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .groups(groups)
                    .bias(bias);
                
                switch (padding_selector) {
                    case 0: opts.padding_mode(torch::kZeros); break;
                    case 1: opts.padding_mode(torch::kReflect); break;
                    case 2: opts.padding_mode(torch::kReplicate); break;
                    case 3: opts.padding_mode(torch::kCircular); break;
                }
                
                torch::nn::Conv1d conv2(opts);
                torch::Tensor output3 = conv2->forward(input);
            } catch (...) {
                // Silently ignore padding mode incompatibilities
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