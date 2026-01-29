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
        // Need sufficient bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters from fuzzer data
        // Extract output_size
        int64_t output_height = (Data[offset++] % 50) + 4;  // 4-53
        int64_t output_width = (Data[offset++] % 50) + 4;   // 4-53
        
        // Extract kernel_size (must be <= output_size)
        int64_t kernel_height = (Data[offset++] % 5) + 1;   // 1-5
        int64_t kernel_width = (Data[offset++] % 5) + 1;    // 1-5
        
        // Extract stride
        int64_t stride_height = (Data[offset++] % 3) + 1;   // 1-3
        int64_t stride_width = (Data[offset++] % 3) + 1;    // 1-3
        
        // Extract padding (must satisfy constraints)
        int64_t padding_height = Data[offset++] % 3;        // 0-2
        int64_t padding_width = Data[offset++] % 3;         // 0-2
        
        // Extract dilation
        int64_t dilation_height = (Data[offset++] % 2) + 1; // 1-2
        int64_t dilation_width = (Data[offset++] % 2) + 1;  // 1-2
        
        // Calculate effective kernel size with dilation
        int64_t eff_kernel_h = dilation_height * (kernel_height - 1) + 1;
        int64_t eff_kernel_w = dilation_width * (kernel_width - 1) + 1;
        
        // Ensure output size is valid for the kernel
        if (output_height + 2 * padding_height < eff_kernel_h) {
            output_height = eff_kernel_h - 2 * padding_height + 1;
        }
        if (output_width + 2 * padding_width < eff_kernel_w) {
            output_width = eff_kernel_w - 2 * padding_width + 1;
        }
        
        // Calculate L (number of sliding blocks)
        int64_t L_height = (output_height + 2 * padding_height - eff_kernel_h) / stride_height + 1;
        int64_t L_width = (output_width + 2 * padding_width - eff_kernel_w) / stride_width + 1;
        int64_t L = L_height * L_width;
        
        if (L <= 0) {
            return 0;
        }
        
        // Extract batch size and channels
        int64_t batch_size = (Data[offset++] % 4) + 1;      // 1-4
        int64_t channels = (Data[offset++] % 4) + 1;        // 1-4
        
        // Input shape for Fold: (N, C * kernel_height * kernel_width, L)
        int64_t input_channels = channels * kernel_height * kernel_width;
        
        // Create properly shaped input tensor
        torch::Tensor input = torch::randn({batch_size, input_channels, L});
        
        // Create fold options: FoldOptions(output_size, kernel_size)
        torch::nn::FoldOptions options(
            {output_height, output_width},
            {kernel_height, kernel_width}
        );
        
        options.stride({stride_height, stride_width})
               .padding({padding_height, padding_width})
               .dilation({dilation_height, dilation_width});
        
        // Create fold module
        torch::nn::Fold fold_module(options);
        
        // Apply fold operation
        torch::Tensor output = fold_module->forward(input);
        
        // Verify output shape: should be (N, C, output_height, output_width)
        auto output_sizes = output.sizes();
        if (output_sizes.size() != 4) {
            std::cerr << "Unexpected output dimensions" << std::endl;
            return -1;
        }
        
        // Access some values to ensure computation completes
        (void)output.sum().item<float>();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}