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
        
        // Need at least some data to proceed
        if (Size < 20) {
            return 0;
        }
        
        // Parse kernel size (3 values for 3D)
        std::vector<int64_t> kernel_size;
        for (int i = 0; i < 3 && offset + 1 <= Size; i++) {
            int64_t k = (Data[offset++] % 4) + 1;  // kernel size 1-4
            kernel_size.push_back(k);
        }
        while (kernel_size.size() < 3) {
            kernel_size.push_back(2);
        }
        
        // Parse stride
        std::vector<int64_t> stride;
        for (int i = 0; i < 3 && offset + 1 <= Size; i++) {
            int64_t s = (Data[offset++] % 3) + 1;  // stride 1-3
            stride.push_back(s);
        }
        while (stride.size() < 3) {
            stride.push_back(kernel_size[stride.size()]);
        }
        
        // Parse padding
        std::vector<int64_t> padding;
        for (int i = 0; i < 3 && offset + 1 <= Size; i++) {
            int64_t p = Data[offset++] % 3;  // padding 0-2
            // Padding must be less than or equal to half of kernel size
            p = std::min(p, kernel_size[i] / 2);
            padding.push_back(p);
        }
        while (padding.size() < 3) {
            padding.push_back(0);
        }
        
        // Parse input dimensions for 5D tensor (N, C, D, H, W)
        int64_t batch_size = (offset < Size) ? (Data[offset++] % 4) + 1 : 1;
        int64_t channels = (offset < Size) ? (Data[offset++] % 4) + 1 : 1;
        int64_t depth = (offset < Size) ? (Data[offset++] % 8) + 2 : 4;
        int64_t height = (offset < Size) ? (Data[offset++] % 8) + 2 : 4;
        int64_t width = (offset < Size) ? (Data[offset++] % 8) + 2 : 4;
        
        // Create MaxPool3d to get valid pooled output and indices
        torch::nn::MaxPool3d pool(
            torch::nn::MaxPool3dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
        );
        
        // Create input tensor with proper 5D shape for pooling
        torch::Tensor original_input = torch::randn({batch_size, channels, depth, height, width});
        
        // Use random data from fuzzer to modify the tensor values
        if (offset < Size) {
            auto accessor = original_input.accessor<float, 5>();
            size_t total_elements = batch_size * channels * depth * height * width;
            for (size_t i = 0; i < total_elements && offset < Size; i++) {
                int b = i / (channels * depth * height * width);
                int c = (i / (depth * height * width)) % channels;
                int d = (i / (height * width)) % depth;
                int h = (i / width) % height;
                int w = i % width;
                accessor[b][c][d][h][w] = static_cast<float>(Data[offset++]) / 128.0f - 1.0f;
            }
        }
        
        // Inner try-catch for expected failures (shape mismatches, etc.)
        try {
            // Apply MaxPool3d with forward_with_indices to get pooled output and valid indices
            auto pool_result = pool->forward_with_indices(original_input);
            torch::Tensor pooled = std::get<0>(pool_result);
            torch::Tensor indices = std::get<1>(pool_result);
            
            // Create MaxUnpool3d module with same parameters
            torch::nn::MaxUnpool3d unpool(
                torch::nn::MaxUnpool3dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
            );
            
            // Apply MaxUnpool3d with indices from MaxPool3d
            torch::Tensor output;
            
            // Determine whether to use output_size
            bool use_output_size = (offset < Size) && (Data[offset++] % 2 == 0);
            
            if (use_output_size) {
                // Use the original input size as output_size
                std::vector<int64_t> output_size = {depth, height, width};
                output = unpool->forward(pooled, indices, output_size);
            } else {
                output = unpool->forward(pooled, indices);
            }
            
            // Use the output to prevent it from being optimized away
            if (output.defined()) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        }
        catch (const c10::Error &e) {
            // Expected failures (shape mismatches, invalid configurations)
            // Silently ignore
        }
        catch (const std::runtime_error &e) {
            // Expected runtime errors
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}