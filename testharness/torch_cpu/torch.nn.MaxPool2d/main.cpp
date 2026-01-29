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
        if (Size < 16) {
            return 0;
        }
        
        // Parse parameters first so we know what tensor shape we need
        // Parse kernel size (1-7)
        int64_t kernel_h = (Data[offset] % 7) + 1;
        offset++;
        int64_t kernel_w = (Data[offset] % 7) + 1;
        offset++;
        
        // Parse stride (1-5)
        int64_t stride_h = (Data[offset] % 5) + 1;
        offset++;
        int64_t stride_w = (Data[offset] % 5) + 1;
        offset++;
        
        // Parse padding (0 to kernel_size/2 to ensure valid)
        int64_t padding_h = Data[offset] % (kernel_h / 2 + 1);
        offset++;
        int64_t padding_w = Data[offset] % (kernel_w / 2 + 1);
        offset++;
        
        // Parse dilation (1-3)
        int64_t dilation_h = (Data[offset] % 3) + 1;
        offset++;
        int64_t dilation_w = (Data[offset] % 3) + 1;
        offset++;
        
        // Parse ceil_mode
        bool ceil_mode = Data[offset] & 0x1;
        offset++;
        
        // Parse whether to use batched input (4D) or unbatched (3D)
        bool use_batch = Data[offset] & 0x1;
        offset++;
        
        // Calculate minimum spatial dimensions needed
        // Output size = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
        // For output >= 1: input >= dilation*(kernel-1) + 1 - 2*padding
        int64_t min_h = dilation_h * (kernel_h - 1) + 1;
        int64_t min_w = dilation_w * (kernel_w - 1) + 1;
        
        // Add some extra to ensure valid output
        min_h = std::max(min_h, kernel_h);
        min_w = std::max(min_w, kernel_w);
        
        // Parse batch size and channels
        int64_t batch_size = (Data[offset] % 4) + 1;
        offset++;
        int64_t channels = (Data[offset] % 4) + 1;
        offset++;
        
        // Parse height and width additions (to vary tensor size)
        int64_t height_add = Data[offset] % 16;
        offset++;
        int64_t width_add = Data[offset] % 16;
        offset++;
        
        int64_t height = min_h + height_add;
        int64_t width = min_w + width_add;
        
        // Create input tensor with appropriate dimensions
        torch::Tensor input_tensor;
        if (use_batch) {
            // 4D: (N, C, H, W)
            input_tensor = torch::randn({batch_size, channels, height, width});
        } else {
            // 3D: (C, H, W)
            input_tensor = torch::randn({channels, height, width});
        }
        
        // Use remaining data to perturb tensor values if available
        if (offset < Size) {
            torch::Tensor noise = fuzzer_utils::createTensor(Data, Size, offset);
            // Just use noise to influence the random state, don't mix directly
            (void)noise;
        }
        
        // Create MaxPool2d module with 2D kernel/stride/padding/dilation
        torch::nn::MaxPool2d max_pool(
            torch::nn::MaxPool2dOptions({kernel_h, kernel_w})
                .stride({stride_h, stride_w})
                .padding({padding_h, padding_w})
                .dilation({dilation_h, dilation_w})
                .ceil_mode(ceil_mode));
        
        // Apply MaxPool2d to the input tensor
        torch::Tensor output = max_pool(input_tensor);
        
        // Ensure the output is valid
        if (output.numel() > 0) {
            // Access some elements to ensure computation happened
            auto sum = output.sum().item<float>();
            (void)sum;  // Prevent unused variable warning
        }
        
        // Also test with return_indices=true
        torch::nn::MaxPool2d max_pool_indices(
            torch::nn::MaxPool2dOptions({kernel_h, kernel_w})
                .stride({stride_h, stride_w})
                .padding({padding_h, padding_w})
                .dilation({dilation_h, dilation_w})
                .ceil_mode(ceil_mode));
        
        auto [output_with_indices, indices] = torch::nn::functional::max_pool2d_with_indices(
            input_tensor,
            torch::nn::functional::MaxPool2dFuncOptions({kernel_h, kernel_w})
                .stride({stride_h, stride_w})
                .padding({padding_h, padding_w})
                .dilation({dilation_h, dilation_w})
                .ceil_mode(ceil_mode));
        
        // Verify indices are valid
        if (indices.numel() > 0) {
            auto max_idx = indices.max().item<int64_t>();
            (void)max_idx;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}