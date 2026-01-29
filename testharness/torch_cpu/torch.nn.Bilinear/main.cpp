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
        
        // Need at least a few bytes for basic parameters
        if (Size < 12) {
            return 0;
        }
        
        // Extract parameters from fuzzer data first
        uint8_t in1_features_raw = Data[offset++];
        uint8_t in2_features_raw = Data[offset++];
        uint8_t out_features_raw = Data[offset++];
        bool bias = Data[offset++] & 0x1;
        
        // Ensure features are in reasonable range [1, 32]
        int64_t in1_features = (in1_features_raw % 32) + 1;
        int64_t in2_features = (in2_features_raw % 32) + 1;
        int64_t out_features = (out_features_raw % 32) + 1;
        
        // Extract batch size from data
        uint8_t batch_size_raw = Data[offset++];
        int64_t batch_size = (batch_size_raw % 16) + 1;
        
        // Create input tensors with correct shapes for Bilinear
        // Bilinear expects inputs of shape (batch, in1_features) and (batch, in2_features)
        torch::Tensor input1 = torch::randn({batch_size, in1_features});
        torch::Tensor input2 = torch::randn({batch_size, in2_features});
        
        // Use remaining fuzzer data to influence tensor values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t input1_elems = static_cast<size_t>(batch_size * in1_features);
            size_t input2_elems = static_cast<size_t>(batch_size * in2_features);
            
            // Modify some values based on fuzzer input
            auto input1_accessor = input1.accessor<float, 2>();
            auto input2_accessor = input2.accessor<float, 2>();
            
            for (size_t i = 0; i < std::min(remaining, input1_elems) && offset < Size; i++) {
                int64_t row = i / in1_features;
                int64_t col = i % in1_features;
                // Scale byte to float range
                input1_accessor[row][col] = static_cast<float>(Data[offset++] - 128) / 32.0f;
            }
            
            for (size_t i = 0; i < std::min(Size - offset, input2_elems) && offset < Size; i++) {
                int64_t row = i / in2_features;
                int64_t col = i % in2_features;
                input2_accessor[row][col] = static_cast<float>(Data[offset++] - 128) / 32.0f;
            }
        }
        
        // Create the Bilinear module
        torch::nn::Bilinear bilinear(
            torch::nn::BilinearOptions(in1_features, in2_features, out_features).bias(bias)
        );
        
        // Inner try-catch for expected shape/type mismatches
        try {
            // Apply the Bilinear module
            torch::Tensor output = bilinear->forward(input1, input2);
            
            // Verify output shape
            if (output.dim() != 2 || output.size(0) != batch_size || output.size(1) != out_features) {
                // Unexpected output shape - this shouldn't happen with correct inputs
            }
            
            // Perform some operations on the output to ensure it's used
            auto sum = output.sum();
            auto mean = output.mean();
            auto max_val = output.max();
            auto min_val = output.min();
            
            // Convert to CPU scalar to force computation
            float sum_result = sum.item<float>();
            float mean_result = mean.item<float>();
            
            // Use the result in a way that prevents the compiler from optimizing it away
            if (std::isnan(sum_result) || std::isinf(sum_result)) {
                return 0;
            }
            
            // Test with different input configurations
            // Test with 3D input (extra batch dimension)
            if (batch_size > 1) {
                try {
                    torch::Tensor input1_3d = input1.unsqueeze(0); // (1, batch, in1_features)
                    torch::Tensor input2_3d = input2.unsqueeze(0); // (1, batch, in2_features)
                    torch::Tensor output_3d = bilinear->forward(input1_3d, input2_3d);
                    (void)output_3d.sum().item<float>();
                } catch (...) {
                    // Shape mismatch expected in some cases
                }
            }
            
        } catch (const c10::Error &e) {
            // Expected PyTorch errors (shape mismatches, etc.) - catch silently
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}