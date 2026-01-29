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
        
        // Need at least a few bytes for tensor creation and padding values
        if (Size < 8) {
            return 0;
        }
        
        // Extract padding values first
        int64_t padding_left = 0;
        int64_t padding_right = 0;
        
        if (offset + sizeof(int8_t) <= Size) {
            padding_left = std::abs(static_cast<int8_t>(Data[offset])) % 10;
            offset += sizeof(int8_t);
        }
        
        if (offset + sizeof(int8_t) <= Size) {
            padding_right = std::abs(static_cast<int8_t>(Data[offset])) % 10;
            offset += sizeof(int8_t);
        }
        
        // Extract dimensions for the input tensor
        // ReplicationPad1d expects 2D (unbatched: C x W) or 3D (batched: N x C x W) input
        int64_t batch_size = 1;
        int64_t channels = 1;
        int64_t width = 4;
        
        if (offset + 3 <= Size) {
            batch_size = (Data[offset] % 4) + 1;      // 1-4
            channels = (Data[offset + 1] % 4) + 1;    // 1-4
            width = (Data[offset + 2] % 16) + 1;      // 1-16
            offset += 3;
        }
        
        // Create input tensor with appropriate shape for ReplicationPad1d
        // Use remaining data to fill tensor values
        torch::Tensor input;
        bool use_3d = (Size % 2 == 0);
        
        if (use_3d) {
            // 3D input: (N, C, W)
            input = torch::randn({batch_size, channels, width});
        } else {
            // 2D input: (C, W)
            input = torch::randn({channels, width});
        }
        
        // Override some values from fuzzer data if available
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = std::min(remaining / sizeof(float), static_cast<size_t>(input.numel()));
            if (num_elements > 0) {
                auto flat = input.flatten();
                auto accessor = flat.accessor<float, 1>();
                for (size_t i = 0; i < num_elements; i++) {
                    float val;
                    std::memcpy(&val, Data + offset + i * sizeof(float), sizeof(float));
                    // Sanitize the value to avoid NaN/Inf issues
                    if (std::isnan(val) || std::isinf(val)) {
                        val = 0.0f;
                    }
                    accessor[i] = val;
                }
            }
        }
        
        // Create padding configuration
        std::vector<int64_t> padding;
        
        // Try different padding configurations based on fuzzer data
        if (Size % 3 == 0) {
            // Single value padding (applied to both sides)
            padding = {padding_left};
        } else {
            // Two value padding (left, right)
            padding = {padding_left, padding_right};
        }
        
        // Create ReplicationPad1d module with padding options
        torch::nn::ReplicationPad1dOptions options(padding);
        torch::nn::ReplicationPad1d pad_module(options);
        
        // Apply ReplicationPad1d
        torch::Tensor output = pad_module->forward(input);
        
        // Verify output by accessing elements
        if (output.numel() > 0) {
            // Sum to force computation
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
}