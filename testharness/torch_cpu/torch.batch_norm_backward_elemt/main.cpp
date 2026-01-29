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
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract dimensions from fuzzer data
        int64_t batch_size = 1 + (Data[offset++] % 8);      // 1-8
        int64_t num_channels = 1 + (Data[offset++] % 16);   // 1-16
        int64_t height = 1 + (Data[offset++] % 8);          // 1-8
        int64_t width = 1 + (Data[offset++] % 8);           // 1-8
        
        // Determine dtype (float or double)
        bool use_double = (Data[offset++] & 0x1);
        auto dtype = use_double ? torch::kFloat64 : torch::kFloat32;
        
        // Determine if weight is used
        bool use_weight = (Data[offset++] & 0x1);
        
        try
        {
            // Create input tensors with proper shapes for batch norm
            // grad_out and input: [N, C, H, W]
            torch::Tensor grad_out = torch::randn({batch_size, num_channels, height, width}, dtype);
            torch::Tensor input = torch::randn({batch_size, num_channels, height, width}, dtype);
            
            // Statistics tensors: [C]
            torch::Tensor mean = torch::randn({num_channels}, dtype);
            torch::Tensor invstd = torch::rand({num_channels}, dtype).abs() + 0.001;  // Must be positive
            torch::Tensor sum_dy = torch::randn({num_channels}, dtype);
            torch::Tensor sum_dy_xmu = torch::randn({num_channels}, dtype);
            
            // count: integer tensor representing total count across batch dimensions
            int64_t total_count = batch_size * height * width;
            torch::Tensor count = torch::tensor({total_count}, torch::kInt32);
            
            // Optional weight tensor: [C]
            torch::Tensor weight;
            if (use_weight) {
                weight = torch::randn({num_channels}, dtype);
            }
            
            // Apply batch_norm_backward_elemt operation
            torch::Tensor result;
            if (use_weight) {
                result = torch::batch_norm_backward_elemt(
                    grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
            } else {
                result = torch::batch_norm_backward_elemt(
                    grad_out, input, mean, invstd, {}, sum_dy, sum_dy_xmu, count);
            }
            
            // Ensure result is computed
            auto sum = result.sum();
            (void)sum;
        }
        catch (const c10::Error &e)
        {
            // Silently catch shape/type mismatches
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}