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
        
        // Need at least some data to proceed
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters for MaxUnpool2d first
        int64_t kernel_size_h = static_cast<int64_t>(Data[offset++]) % 4 + 1;  // 1-4
        int64_t kernel_size_w = static_cast<int64_t>(Data[offset++]) % 4 + 1;  // 1-4
        int64_t stride_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;       // 1-3
        int64_t stride_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;       // 1-3
        int64_t padding_h = static_cast<int64_t>(Data[offset++]) % 2;          // 0-1
        int64_t padding_w = static_cast<int64_t>(Data[offset++]) % 2;          // 0-1
        
        // Batch size and channels
        int64_t batch_size = static_cast<int64_t>(Data[offset++]) % 4 + 1;     // 1-4
        int64_t channels = static_cast<int64_t>(Data[offset++]) % 8 + 1;       // 1-8
        
        // Pooled (input) height and width - these are the output sizes from MaxPool2d
        int64_t pooled_h = static_cast<int64_t>(Data[offset++]) % 8 + 1;       // 1-8
        int64_t pooled_w = static_cast<int64_t>(Data[offset++]) % 8 + 1;       // 1-8
        
        // Calculate output size (the original input size before pooling)
        // Formula: output_size = (pooled_size - 1) * stride - 2 * padding + kernel_size
        int64_t output_h = (pooled_h - 1) * stride_h - 2 * padding_h + kernel_size_h;
        int64_t output_w = (pooled_w - 1) * stride_w - 2 * padding_w + kernel_size_w;
        
        // Ensure valid output dimensions
        if (output_h <= 0 || output_w <= 0) {
            return 0;
        }
        
        // Create MaxUnpool2d module
        torch::nn::MaxUnpool2d unpool(
            torch::nn::MaxUnpool2dOptions({kernel_size_h, kernel_size_w})
                .stride({stride_h, stride_w})
                .padding({padding_h, padding_w})
        );
        
        // Create input tensor (pooled values) - 4D: (N, C, H, W)
        torch::Tensor input = torch::randn({batch_size, channels, pooled_h, pooled_w});
        
        // Create valid indices tensor
        // Indices should be in range [0, output_h * output_w)
        int64_t max_idx = output_h * output_w;
        torch::Tensor indices = torch::randint(0, max_idx, {batch_size, channels, pooled_h, pooled_w}, torch::kLong);
        
        // Use remaining fuzzer data to modify tensors
        if (offset < Size) {
            size_t remaining = Size - offset;
            int64_t num_elements = input.numel();
            for (size_t i = 0; i < std::min(remaining, static_cast<size_t>(num_elements)); i++) {
                int64_t flat_idx = i % num_elements;
                float val = static_cast<float>(Data[offset + i]) / 25.5f - 5.0f;
                input.view(-1)[flat_idx] = val;
            }
        }
        
        // Test different calling patterns
        int call_mode = (Size > 0) ? Data[Size - 1] % 3 : 0;
        
        try {
            torch::Tensor output;
            
            if (call_mode == 0) {
                // Call with explicit output_size
                std::vector<int64_t> output_size = {batch_size, channels, output_h, output_w};
                output = unpool->forward(input, indices, output_size);
            } else if (call_mode == 1) {
                // Call without output_size - relies on kernel/stride/padding to infer
                output = unpool->forward(input, indices);
            } else {
                // Test with slightly different output size (may fail, which is expected)
                std::vector<int64_t> output_size = {batch_size, channels, output_h + 1, output_w + 1};
                output = unpool->forward(input, indices, output_size);
            }
            
            // Verify output
            auto output_sizes = output.sizes();
            (void)output_sizes;
            
            // Perform some operations to exercise the output
            auto sum = output.sum();
            (void)sum;
        }
        catch (const c10::Error& e) {
            // Expected errors for invalid configurations (shape mismatches, etc.)
        }
        catch (const std::runtime_error& e) {
            // Expected runtime errors for invalid parameters
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
}