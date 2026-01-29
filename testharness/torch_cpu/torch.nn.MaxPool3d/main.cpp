#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes for basic tensor creation and parameters
        if (Size < 10) {
            return 0;
        }
        
        // Extract parameters for MaxPool3d from the data first
        int64_t kernel_size = static_cast<int64_t>(Data[offset++]) % 5 + 1;  // 1-5
        int64_t stride = static_cast<int64_t>(Data[offset++]) % 4 + 1;       // 1-4
        int64_t padding = static_cast<int64_t>(Data[offset++]) % 3;          // 0-2
        int64_t dilation = static_cast<int64_t>(Data[offset++]) % 3 + 1;     // 1-3
        bool ceil_mode = Data[offset++] % 2 == 1;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float for MaxPool3d
        input = input.to(torch::kFloat32);
        
        // Ensure we have a 5D tensor (batch_size, channels, depth, height, width)
        if (input.dim() == 0 || input.numel() == 0) {
            return 0;
        }
        
        // Reshape to 5D if needed
        if (input.dim() < 5) {
            int64_t total_elements = input.numel();
            // Create a valid 5D shape: [1, 1, D, H, W] where D*H*W = total_elements
            // We need spatial dims large enough for the kernel
            int64_t min_spatial = kernel_size + (kernel_size - 1) * (dilation - 1);
            
            // Calculate spatial dimensions
            int64_t spatial_size = std::max(min_spatial, static_cast<int64_t>(std::cbrt(total_elements)));
            spatial_size = std::max(spatial_size, int64_t(2));
            
            // Pad the tensor if needed to fit the new shape
            int64_t needed_elements = spatial_size * spatial_size * spatial_size;
            if (total_elements < needed_elements) {
                // Repeat the tensor to fill the shape
                input = input.flatten();
                while (input.numel() < needed_elements) {
                    input = torch::cat({input, input});
                }
                input = input.slice(0, 0, needed_elements);
            } else {
                input = input.flatten().slice(0, 0, needed_elements);
            }
            
            input = input.reshape({1, 1, spatial_size, spatial_size, spatial_size});
        } else if (input.dim() > 5) {
            // Flatten extra dimensions into batch
            auto sizes = input.sizes().vec();
            int64_t batch = 1;
            for (size_t i = 0; i < sizes.size() - 4; i++) {
                batch *= sizes[i];
            }
            input = input.reshape({batch, sizes[sizes.size()-4], sizes[sizes.size()-3], 
                                   sizes[sizes.size()-2], sizes[sizes.size()-1]});
        }
        
        // Ensure spatial dimensions are large enough for the pooling operation
        int64_t min_size = kernel_size + (kernel_size - 1) * (dilation - 1);
        if (input.size(2) < min_size || input.size(3) < min_size || input.size(4) < min_size) {
            return 0;  // Input too small for this kernel configuration
        }
        
        // Validate padding isn't too large
        if (padding >= (kernel_size + 1) / 2) {
            padding = 0;
        }
        
        // Create MaxPool3d module
        torch::nn::MaxPool3d max_pool(
            torch::nn::MaxPool3dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .ceil_mode(ceil_mode)
        );
        
        // Apply MaxPool3d to the input tensor
        torch::Tensor output;
        try {
            output = max_pool->forward(input);
        } catch (const c10::Error &e) {
            // Shape mismatch or invalid configuration - silently discard
            return 0;
        }
        
        // Verify output is valid
        if (output.numel() > 0) {
            // Force computation by summing
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Also test with return_indices option
        torch::nn::MaxPool3d max_pool_indices(
            torch::nn::MaxPool3dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .ceil_mode(ceil_mode)
        );
        
        try {
            auto result = torch::max_pool3d_with_indices(input, {kernel_size, kernel_size, kernel_size},
                                                          {stride, stride, stride},
                                                          {padding, padding, padding},
                                                          {dilation, dilation, dilation},
                                                          ceil_mode);
            torch::Tensor pooled = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
            if (pooled.numel() > 0) {
                volatile float sum = pooled.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error &e) {
            // Silently handle invalid configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}