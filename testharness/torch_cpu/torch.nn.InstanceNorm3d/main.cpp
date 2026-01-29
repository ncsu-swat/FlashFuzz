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
        
        // Need at least a few bytes for basic parameters
        if (Size < 16) {
            return 0;
        }
        
        // Extract dimensions for 5D tensor (N, C, D, H, W)
        int64_t N = (Data[offset++] % 4) + 1;      // 1-4 batch size
        int64_t C = (Data[offset++] % 32) + 1;    // 1-32 channels
        int64_t D = (Data[offset++] % 8) + 1;     // 1-8 depth
        int64_t H = (Data[offset++] % 8) + 1;     // 1-8 height
        int64_t W = (Data[offset++] % 8) + 1;     // 1-8 width
        
        // Extract other parameters
        bool affine = Data[offset++] & 1;
        bool track_running_stats = Data[offset++] & 1;
        bool eval_mode = Data[offset++] & 1;
        bool test_backward = Data[offset++] & 1;
        
        // Extract eps parameter (small positive value for numerical stability)
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is positive and reasonable
            eps_f = std::abs(eps_f);
            if (!std::isfinite(eps_f) || eps_f == 0.0f) {
                eps = 1e-5;
            } else if (eps_f > 1.0f) {
                eps = 1.0;
            } else if (eps_f < 1e-10f) {
                eps = 1e-5;
            } else {
                eps = static_cast<double>(eps_f);
            }
        }
        
        // Extract momentum parameter
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float momentum_f;
            std::memcpy(&momentum_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure momentum is in [0, 1] and finite
            momentum_f = std::abs(momentum_f);
            if (!std::isfinite(momentum_f)) {
                momentum = 0.1;
            } else if (momentum_f > 1.0f) {
                momentum = 1.0;
            } else {
                momentum = static_cast<double>(momentum_f);
            }
        }
        
        // Create 5D input tensor with floating point type
        torch::Tensor input = torch::randn({N, C, D, H, W}, torch::kFloat32);
        
        // Also try using fuzzed data for tensor values if available
        if (offset + 4 <= Size) {
            // Use remaining fuzzed bytes to create some tensor values
            size_t remaining = Size - offset;
            size_t num_elements = input.numel();
            size_t fill_count = std::min(remaining / sizeof(float), num_elements);
            
            if (fill_count > 0) {
                auto input_accessor = input.accessor<float, 5>();
                size_t idx = 0;
                for (int64_t n = 0; n < N && idx < fill_count; n++) {
                    for (int64_t c = 0; c < C && idx < fill_count; c++) {
                        for (int64_t d = 0; d < D && idx < fill_count; d++) {
                            for (int64_t h = 0; h < H && idx < fill_count; h++) {
                                for (int64_t w = 0; w < W && idx < fill_count; w++) {
                                    float val;
                                    std::memcpy(&val, Data + offset + idx * sizeof(float), sizeof(float));
                                    if (std::isfinite(val)) {
                                        input_accessor[n][c][d][h][w] = val;
                                    }
                                    idx++;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Create InstanceNorm3d module
        torch::nn::InstanceNorm3d instance_norm(
            torch::nn::InstanceNorm3dOptions(C)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Set module mode
        if (eval_mode) {
            instance_norm->eval();
        } else {
            instance_norm->train();
        }
        
        // Apply the module - wrap in try-catch for expected shape/value errors
        torch::Tensor output;
        try {
            output = instance_norm->forward(input);
        } catch (const c10::Error&) {
            // Expected errors from invalid configurations
            return 0;
        }
        
        // Perform a simple operation on the output to ensure it's used
        auto sum = output.sum();
        (void)sum;
        
        // Test backward pass
        if (test_backward) {
            try {
                // Create fresh input requiring gradients
                auto input_with_grad = torch::randn({N, C, D, H, W}, 
                    torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
                
                // Create fresh module for backward test
                torch::nn::InstanceNorm3d instance_norm_back(
                    torch::nn::InstanceNorm3dOptions(C)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats)
                );
                
                instance_norm_back->train();
                
                // Forward pass
                auto output_for_backward = instance_norm_back->forward(input_with_grad);
                
                // Backward pass
                output_for_backward.sum().backward();
                
                // Access gradient to ensure it was computed
                auto grad = input_with_grad.grad();
                (void)grad;
            } catch (const c10::Error&) {
                // Expected errors during backward pass
            }
        }
        
        // Test with different tensor types
        if (offset < Size && (Data[offset] & 1)) {
            try {
                auto input_double = input.to(torch::kFloat64);
                
                torch::nn::InstanceNorm3d instance_norm_double(
                    torch::nn::InstanceNorm3dOptions(C)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats)
                );
                instance_norm_double->to(torch::kFloat64);
                
                auto output_double = instance_norm_double->forward(input_double);
                (void)output_double;
            } catch (const c10::Error&) {
                // Expected for unsupported dtypes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}