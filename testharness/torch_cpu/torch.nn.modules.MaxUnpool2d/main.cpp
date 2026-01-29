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
        
        // Need sufficient data
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters from fuzz data
        uint8_t batch_size = (Data[offset++] % 4) + 1;    // 1-4
        uint8_t channels = (Data[offset++] % 4) + 1;      // 1-4
        uint8_t height = (Data[offset++] % 8) + 2;        // 2-9
        uint8_t width = (Data[offset++] % 8) + 2;         // 2-9
        
        uint8_t kernel_h = (Data[offset++] % 3) + 2;      // 2-4
        uint8_t kernel_w = (Data[offset++] % 3) + 2;      // 2-4
        
        uint8_t stride_h = (Data[offset++] % 2) + 1;      // 1-2
        uint8_t stride_w = (Data[offset++] % 2) + 1;      // 1-2
        
        uint8_t padding_h = Data[offset++] % 2;           // 0-1
        uint8_t padding_w = Data[offset++] % 2;           // 0-1
        
        bool use_output_size = Data[offset++] % 2;
        
        std::vector<int64_t> kernel_size = {kernel_h, kernel_w};
        std::vector<int64_t> stride = {stride_h, stride_w};
        std::vector<int64_t> padding = {padding_h, padding_w};
        
        // Create a proper input tensor and run MaxPool2d to get valid indices
        // The pooled output dimensions
        int64_t in_h = height;
        int64_t in_w = width;
        
        // Create a larger tensor to pool from (reverse engineer the input size)
        int64_t orig_h = (in_h - 1) * stride_h - 2 * padding_h + kernel_h;
        int64_t orig_w = (in_w - 1) * stride_w - 2 * padding_w + kernel_w;
        
        if (orig_h <= 0 || orig_w <= 0) {
            return 0;
        }
        
        // Create original tensor for pooling
        torch::Tensor original = torch::randn({batch_size, channels, orig_h, orig_w});
        
        // Use fuzz data to modify tensor values
        if (offset < Size) {
            torch::Tensor flattened = original.flatten();
            size_t num_elements = std::min(flattened.numel(), static_cast<int64_t>(Size - offset));
            auto accessor = flattened.accessor<float, 1>();
            for (size_t i = 0; i < num_elements && offset < Size; i++, offset++) {
                accessor[i] = static_cast<float>(Data[offset]) / 25.5f - 5.0f;
            }
        }
        
        // Run MaxPool2d to get valid indices
        torch::nn::MaxPool2d pool(
            torch::nn::MaxPool2dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
        );
        
        auto [pooled, indices] = pool->forward_with_indices(original);
        
        // Create MaxUnpool2d module with same parameters
        torch::nn::MaxUnpool2d unpool(
            torch::nn::MaxUnpool2dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
        );
        
        // Apply the unpooling operation
        torch::Tensor output;
        
        try {
            if (use_output_size) {
                // Use explicit output size matching original
                std::vector<int64_t> out_size = {batch_size, channels, orig_h, orig_w};
                output = unpool->forward(pooled, indices, out_size);
            } else {
                output = unpool->forward(pooled, indices);
            }
            
            // Verify output
            auto sizes = output.sizes();
            auto dtype = output.dtype();
            
            // Additional operation to exercise the output
            torch::Tensor sum = output.sum();
            (void)sum.item<float>();
        }
        catch (const c10::Error&) {
            // Shape mismatch or other expected errors - silently ignore
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}