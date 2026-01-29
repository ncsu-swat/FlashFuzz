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
        size_t offset = 0;
        
        // Need enough data to proceed
        if (Size < 20) {
            return 0;
        }
        
        // Extract parameters for MaxUnpool3d
        int64_t kernel_size_d = static_cast<int64_t>(Data[offset++]) % 3 + 2;  // 2-4
        int64_t kernel_size_h = static_cast<int64_t>(Data[offset++]) % 3 + 2;  // 2-4
        int64_t kernel_size_w = static_cast<int64_t>(Data[offset++]) % 3 + 2;  // 2-4
        int64_t stride_d = static_cast<int64_t>(Data[offset++]) % 3 + 1;       // 1-3
        int64_t stride_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;       // 1-3
        int64_t stride_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;       // 1-3
        int64_t padding_d = static_cast<int64_t>(Data[offset++]) % 2;          // 0-1
        int64_t padding_h = static_cast<int64_t>(Data[offset++]) % 2;          // 0-1
        int64_t padding_w = static_cast<int64_t>(Data[offset++]) % 2;          // 0-1
        
        // Extract dimensions for input tensor (keep small to avoid memory issues)
        int64_t batch = static_cast<int64_t>(Data[offset++]) % 3 + 1;     // 1-3
        int64_t channels = static_cast<int64_t>(Data[offset++]) % 4 + 1;  // 1-4
        int64_t in_d = static_cast<int64_t>(Data[offset++]) % 4 + 2;      // 2-5
        int64_t in_h = static_cast<int64_t>(Data[offset++]) % 4 + 2;      // 2-5
        int64_t in_w = static_cast<int64_t>(Data[offset++]) % 4 + 2;      // 2-5
        
        // Calculate expected output sizes for MaxUnpool3d
        // output_size = (input_size - 1) * stride - 2 * padding + kernel_size
        int64_t out_d = (in_d - 1) * stride_d - 2 * padding_d + kernel_size_d;
        int64_t out_h = (in_h - 1) * stride_h - 2 * padding_h + kernel_size_h;
        int64_t out_w = (in_w - 1) * stride_w - 2 * padding_w + kernel_size_w;
        
        // Ensure output dimensions are positive
        if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
            return 0;
        }
        
        // Create proper 5D input tensor for MaxUnpool3d (N, C, D, H, W)
        torch::Tensor input = torch::randn({batch, channels, in_d, in_h, in_w});
        
        // To get proper indices, we should run MaxPool3d first
        // This ensures indices are valid for the unpool operation
        torch::nn::MaxPool3d pool(
            torch::nn::MaxPool3dOptions({kernel_size_d, kernel_size_h, kernel_size_w})
                .stride({stride_d, stride_h, stride_w})
                .padding({padding_d, padding_h, padding_w})
        );
        
        // Create a tensor of the output size and pool it to get valid indices
        torch::Tensor large_input = torch::randn({batch, channels, out_d, out_h, out_w});
        
        // Use fuzz data to influence the values
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 25.5f - 5.0f;
            large_input = large_input * scale;
        }
        
        // Run MaxPool3d with return_indices=true to get valid indices
        auto pool_result = torch::max_pool3d_with_indices(
            large_input,
            {kernel_size_d, kernel_size_h, kernel_size_w},
            {stride_d, stride_h, stride_w},
            {padding_d, padding_h, padding_w}
        );
        
        torch::Tensor pooled = std::get<0>(pool_result);
        torch::Tensor indices = std::get<1>(pool_result);
        
        // Create MaxUnpool3d module
        torch::nn::MaxUnpool3d unpool(
            torch::nn::MaxUnpool3dOptions({kernel_size_d, kernel_size_h, kernel_size_w})
                .stride({stride_d, stride_h, stride_w})
                .padding({padding_d, padding_h, padding_w})
        );
        
        // Apply MaxUnpool3d - pooled output becomes unpool input
        torch::Tensor output;
        
        // Test with explicit output_size
        bool use_output_size = (offset < Size) && (Data[offset++] % 2 == 0);
        
        try {
            if (use_output_size) {
                std::vector<int64_t> output_size = {batch, channels, out_d, out_h, out_w};
                output = unpool->forward(pooled, indices, output_size);
            } else {
                output = unpool->forward(pooled, indices);
            }
            
            // Use the output to prevent optimization
            if (output.defined()) {
                volatile auto sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error &e) {
            // Inner catch for expected shape/size mismatches
            // Don't log these as they're expected during fuzzing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}