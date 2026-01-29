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
        // Need minimum data for dimension parameters
        if (Size < 12) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions from fuzzer data
        int64_t batch_size = 1 + (Data[offset++] % 8);      // 1-8
        int64_t in_features1 = 1 + (Data[offset++] % 16);   // 1-16
        int64_t in_features2 = 1 + (Data[offset++] % 16);   // 1-16
        int64_t out_features = 1 + (Data[offset++] % 16);   // 1-16

        // Determine dtype
        auto dtype_options = torch::TensorOptions().dtype(torch::kFloat32);
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 3;
            if (dtype_choice == 1) {
                dtype_options = torch::TensorOptions().dtype(torch::kFloat64);
            }
            // Keep float32 for other choices for compatibility
        }

        // Create input1: (batch_size, in_features1)
        torch::Tensor input1 = torch::randn({batch_size, in_features1}, dtype_options);

        // Create input2: (batch_size, in_features2)
        torch::Tensor input2 = torch::randn({batch_size, in_features2}, dtype_options);

        // Create weight: (out_features, in_features1, in_features2)
        torch::Tensor weight = torch::randn({out_features, in_features1, in_features2}, dtype_options);

        // Create bias: (out_features)
        torch::Tensor bias = torch::randn({out_features}, dtype_options);

        // Scale tensors based on fuzzer input to get varied values
        if (offset < Size) {
            float scale1 = static_cast<float>(Data[offset++]) / 25.5f;
            input1 = input1 * scale1;
        }
        if (offset < Size) {
            float scale2 = static_cast<float>(Data[offset++]) / 25.5f;
            input2 = input2 * scale2;
        }
        if (offset < Size) {
            float scale_w = static_cast<float>(Data[offset++]) / 25.5f;
            weight = weight * scale_w;
        }

        // Test bilinear with bias
        torch::Tensor result = torch::bilinear(input1, input2, weight, bias);

        // Verify output shape: should be (batch_size, out_features)
        if (result.dim() != 2 || result.size(0) != batch_size || result.size(1) != out_features) {
            std::cerr << "Unexpected output shape" << std::endl;
            return -1;
        }

        // Test bilinear without bias (empty tensor)
        try {
            torch::Tensor result_no_bias = torch::bilinear(input1, input2, weight, torch::Tensor());
            (void)result_no_bias.sum();
        } catch (...) {
            // Some configurations may not support empty bias
        }

        // Test with different batch sizes (3D input)
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            int64_t seq_len = 1 + (Data[offset % Size] % 4);
            torch::Tensor input1_3d = torch::randn({batch_size, seq_len, in_features1}, dtype_options);
            torch::Tensor input2_3d = torch::randn({batch_size, seq_len, in_features2}, dtype_options);
            
            try {
                torch::Tensor result_3d = torch::bilinear(input1_3d, input2_3d, weight, bias);
                (void)result_3d.sum();
            } catch (...) {
                // Shape variations may fail, that's expected
            }
        }

        // Force computation to ensure the operation is fully executed
        auto sum_val = result.sum();
        (void)sum_val.item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}