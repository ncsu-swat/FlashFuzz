#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

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
        
        // Ensure input has at least 3 dimensions for Conv1d (batch_size, channels, length)
        if (input.dim() < 3) {
            // Reshape to make it compatible with Conv1d
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, 1, 1]
                new_shape = {1, 1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, 1, length]
                new_shape = {1, 1, input.size(0)};
            } else if (input.dim() == 2) {
                // 2D tensor, reshape to [batch_size, 1, length]
                new_shape = {input.size(0), 1, input.size(1)};
            }
            
            // Reshape the tensor
            input = input.reshape(new_shape);
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
            out_channels = (Data[offset] % 8) + 1; // 1-8 output channels
            offset++;
            
            kernel_size = (Data[offset] % 5) + 1; // 1-5 kernel size
            offset++;
            
            stride = (Data[offset] % 3) + 1; // 1-3 stride
            offset++;
            
            padding = Data[offset] % 3; // 0-2 padding
            offset++;
            
            dilation = (Data[offset] % 2) + 1; // 1-2 dilation
            offset++;
            
            // Ensure groups divides in_channels
            if (in_channels > 0) {
                groups = (Data[offset] % in_channels) + 1;
                // Ensure groups divides in_channels
                if (in_channels % groups != 0) {
                    groups = 1; // Default to 1 if not divisible
                }
            }
            offset++;
            
            bias = (Data[offset] % 2) == 0; // 50% chance of bias
            offset++;
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
            // Create another tensor with different data type
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure input2 has at least 3 dimensions
            if (input2.dim() < 3) {
                std::vector<int64_t> new_shape;
                if (input2.dim() == 0) {
                    new_shape = {1, in_channels, 1};
                } else if (input2.dim() == 1) {
                    new_shape = {1, in_channels, input2.size(0)};
                } else if (input2.dim() == 2) {
                    new_shape = {input2.size(0), in_channels, input2.size(1)};
                }
                
                // Reshape the tensor
                input2 = input2.reshape(new_shape);
            }
            
            // Ensure input2 has the correct number of channels
            if (input2.size(1) != in_channels) {
                input2 = input2.reshape({input2.size(0), in_channels, -1});
            }
            
            // Apply Conv1d to input2
            torch::Tensor output2 = conv->forward(input2);
        }
        
        // Try with different padding modes if we have more data
        if (offset + 1 < Size) {
            torch::nn::detail::conv_padding_mode_t padding_mode;
            uint8_t padding_selector = Data[offset++] % 4;
            
            switch (padding_selector) {
                case 0: padding_mode = torch::kZeros; break;
                case 1: padding_mode = torch::kReflect; break;
                case 2: padding_mode = torch::kReplicate; break;
                case 3: padding_mode = torch::kCircular; break;
            }
            
            torch::nn::Conv1d conv2(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                    .stride(stride)
                                    .padding(padding)
                                    .dilation(dilation)
                                    .groups(groups)
                                    .bias(bias)
                                    .padding_mode(padding_mode));
            
            torch::Tensor output3 = conv2->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}