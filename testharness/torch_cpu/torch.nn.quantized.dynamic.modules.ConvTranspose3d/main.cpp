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
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (N, C, D, H, W) for ConvTranspose3d
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            int64_t total_elements = input.numel();
            int64_t batch_size = 1;
            int64_t channels = 1;
            int64_t depth = 1;
            int64_t height = 1;
            int64_t width = 1;
            
            if (total_elements > 0) {
                // Distribute elements across dimensions
                width = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(total_elements / 4)));
                height = width;
                depth = std::max(static_cast<int64_t>(1), static_cast<int64_t>(total_elements / (batch_size * channels * width * height)));
            }
            
            // Reshape tensor to 5D
            input = input.reshape({batch_size, channels, depth, height, width});
        }
        
        // Extract parameters for ConvTranspose3d from the remaining data
        int64_t in_channels = input.size(1);
        int64_t out_channels = in_channels;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 1;
        int64_t output_padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool bias = true;
        
        // Use remaining data to set parameters if available
        if (offset + 8 < Size) {
            out_channels = (Data[offset] % 8) + 1;
            kernel_size = (Data[offset + 1] % 5) + 1;
            stride = (Data[offset + 2] % 3) + 1;
            padding = Data[offset + 3] % 3;
            output_padding = Data[offset + 4] % 2;
            dilation = (Data[offset + 5] % 2) + 1;
            groups = std::gcd(in_channels, out_channels);
            if (groups > 1) {
                groups = (Data[offset + 6] % groups) + 1;
            }
            bias = (Data[offset + 7] % 2) == 0;
            offset += 8;
        }
        
        // Ensure parameters are valid
        if (in_channels % groups != 0 || out_channels % groups != 0) {
            groups = 1;
        }
        
        // Create the ConvTranspose3d module
        torch::nn::ConvTranspose3dOptions options(in_channels, out_channels, kernel_size);
        options.stride(stride)
               .padding(padding)
               .output_padding(output_padding)
               .dilation(dilation)
               .groups(groups)
               .bias(bias);
        
        auto conv_module = torch::nn::ConvTranspose3d(options);
        
        // Apply the module to the input tensor
        torch::Tensor output = conv_module->forward(input);
        
        // Quantize the output tensor dynamically
        torch::Tensor quantized_output = torch::quantize_per_tensor(
            output, 
            0.1, // scale
            0,   // zero_point
            torch::kQUInt8
        );
        
        // Try different weight scales
        if (offset + 1 < Size) {
            float scale = (Data[offset] % 100) / 100.0f + 0.01f;
            int zero_point = 0;
            
            // Re-quantize with new scale
            quantized_output = torch::quantize_per_tensor(
                output, 
                scale, 
                zero_point, 
                torch::kQUInt8
            );
        }
        
        // Try different configurations
        if (offset + 2 < Size) {
            // Try different dtype
            auto dtype = fuzzer_utils::parseDataType(Data[offset]);
            if (dtype == torch::kFloat || dtype == torch::kDouble || dtype == torch::kHalf) {
                input = input.to(dtype);
                output = conv_module->forward(input);
            }
            
            // Try different batch size
            if (offset + 3 < Size) {
                int64_t new_batch_size = (Data[offset + 1] % 4) + 1;
                if (new_batch_size != input.size(0) && input.numel() > 0) {
                    auto shape = input.sizes().vec();
                    shape[0] = new_batch_size;
                    int64_t total_elements = 1;
                    for (auto dim : shape) {
                        total_elements *= dim;
                    }
                    
                    if (total_elements > 0) {
                        input = input.expand({new_batch_size, -1, -1, -1, -1});
                        output = conv_module->forward(input);
                    }
                }
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
