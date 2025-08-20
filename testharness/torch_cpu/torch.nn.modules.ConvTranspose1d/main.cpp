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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (batch_size, channels, length)
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvTranspose1d from the remaining data
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel_size = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Get in_channels from input tensor
        in_channels = input.size(1);
        
        // Parse remaining parameters from data
        if (offset + 8 <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_channels is positive
            out_channels = std::abs(out_channels) % 16 + 1;
        } else {
            out_channels = 1;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&kernel_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure kernel_size is positive
            kernel_size = std::abs(kernel_size) % 7 + 1;
        } else {
            kernel_size = 3;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&stride, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure stride is positive
            stride = std::abs(stride) % 4 + 1;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow padding to be any value
            padding = padding % 5;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&output_padding, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_padding is non-negative and less than stride
            output_padding = std::abs(output_padding) % stride;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&dilation, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure dilation is positive
            dilation = std::abs(dilation) % 3 + 1;
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure groups is positive and a divisor of in_channels
            groups = std::abs(groups) % in_channels + 1;
            if (in_channels % groups != 0) {
                groups = 1;  // Default to 1 if not a divisor
            }
        }
        
        if (offset < Size) {
            bias = Data[offset] & 1;  // Use lowest bit to determine bias
        }
        
        // Create ConvTranspose1d module
        torch::nn::ConvTranspose1dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride);
        options.padding(padding);
        options.output_padding(output_padding);
        options.dilation(dilation);
        options.groups(groups);
        options.bias(bias);
        
        torch::nn::ConvTranspose1d conv_transpose(options);
        
        // Apply the convolution
        torch::Tensor output = conv_transpose->forward(input);
        
        // Try different input types
        if (offset + 1 < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset]);
            if (dtype != input.scalar_type()) {
                try {
                    torch::Tensor input_cast = input.to(dtype);
                    torch::Tensor output_cast = conv_transpose->forward(input_cast);
                } catch (const std::exception &) {
                    // Ignore exceptions from type conversion
                }
            }
        }
        
        // Try with different batch sizes
        if (input.size(0) > 1 && input.size(0) % 2 == 0) {
            try {
                torch::Tensor half_batch = input.slice(0, 0, input.size(0) / 2);
                torch::Tensor output_half = conv_transpose->forward(half_batch);
            } catch (const std::exception &) {
                // Ignore exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}