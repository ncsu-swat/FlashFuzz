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
        
        // Ensure input has 5 dimensions (batch_size, channels, depth, height, width)
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            int64_t total_elements = input.numel();
            int64_t batch_size = 1;
            int64_t in_channels = 1;
            int64_t depth = 1;
            int64_t height = 1;
            int64_t width = 1;
            
            if (total_elements > 0) {
                // Distribute elements across dimensions
                width = std::min(static_cast<int64_t>(4), total_elements);
                total_elements /= width;
                
                if (total_elements > 0) {
                    height = std::min(static_cast<int64_t>(4), total_elements);
                    total_elements /= height;
                    
                    if (total_elements > 0) {
                        depth = std::min(static_cast<int64_t>(4), total_elements);
                        total_elements /= depth;
                        
                        if (total_elements > 0) {
                            in_channels = std::min(static_cast<int64_t>(3), total_elements);
                            total_elements /= in_channels;
                            
                            if (total_elements > 0) {
                                batch_size = total_elements;
                            }
                        }
                    }
                }
            }
            
            input = input.reshape({batch_size, in_channels, depth, height, width});
        }
        
        // Extract parameters for ConvTranspose3d from the remaining data
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Parse out_channels
        uint16_t out_channels = 1;
        if (offset + 2 <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            out_channels = (out_channels % 8) + 1; // Limit to reasonable range
        }
        
        // Parse kernel_size
        uint16_t kernel_d = 1, kernel_h = 1, kernel_w = 1;
        if (offset + 6 <= Size) {
            std::memcpy(&kernel_d, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            std::memcpy(&kernel_h, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            std::memcpy(&kernel_w, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            
            kernel_d = (kernel_d % 5) + 1; // Limit to reasonable range
            kernel_h = (kernel_h % 5) + 1;
            kernel_w = (kernel_w % 5) + 1;
        }
        
        // Parse stride
        uint8_t stride_d = 1, stride_h = 1, stride_w = 1;
        if (offset + 3 <= Size) {
            stride_d = (Data[offset++] % 3) + 1;
            stride_h = (Data[offset++] % 3) + 1;
            stride_w = (Data[offset++] % 3) + 1;
        }
        
        // Parse padding
        uint8_t padding_d = 0, padding_h = 0, padding_w = 0;
        if (offset + 3 <= Size) {
            padding_d = Data[offset++] % 3;
            padding_h = Data[offset++] % 3;
            padding_w = Data[offset++] % 3;
        }
        
        // Parse output_padding
        uint8_t output_padding_d = 0, output_padding_h = 0, output_padding_w = 0;
        if (offset + 3 <= Size) {
            output_padding_d = Data[offset++] % 2;
            output_padding_h = Data[offset++] % 2;
            output_padding_w = Data[offset++] % 2;
        }
        
        // Parse groups
        uint8_t groups = 1;
        if (offset < Size) {
            groups = (Data[offset++] % 4) + 1;
        }
        
        // Parse bias flag
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] % 2 == 0;
        }
        
        // Parse dilation
        uint8_t dilation_d = 1, dilation_h = 1, dilation_w = 1;
        if (offset + 3 <= Size) {
            dilation_d = (Data[offset++] % 2) + 1;
            dilation_h = (Data[offset++] % 2) + 1;
            dilation_w = (Data[offset++] % 2) + 1;
        }
        
        // Get input channels from the input tensor
        int64_t in_channels = input.size(1);
        
        // Ensure groups divides both in_channels and out_channels
        if (in_channels % groups != 0 || out_channels % groups != 0) {
            groups = 1; // Reset to 1 if not divisible
        }
        
        // Create ConvTranspose3d module
        torch::nn::ConvTranspose3dOptions options(
            in_channels, 
            out_channels, 
            {kernel_d, kernel_h, kernel_w}
        );
        
        options.stride({stride_d, stride_h, stride_w})
               .padding({padding_d, padding_h, padding_w})
               .output_padding({output_padding_d, output_padding_h, output_padding_w})
               .groups(groups)
               .bias(bias)
               .dilation({dilation_d, dilation_h, dilation_w});
        
        torch::nn::ConvTranspose3d conv_transpose = torch::nn::ConvTranspose3d(options);
        
        // Apply the ConvTranspose3d operation
        torch::Tensor output = conv_transpose->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}