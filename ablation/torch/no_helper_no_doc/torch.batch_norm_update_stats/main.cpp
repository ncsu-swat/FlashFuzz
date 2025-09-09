#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for the operation
        if (Size < 32) {
            return 0;
        }

        // Extract basic parameters
        auto batch_size = extract_int(Data, Size, offset, 1, 128);
        auto channels = extract_int(Data, Size, offset, 1, 64);
        auto height = extract_int(Data, Size, offset, 1, 32);
        auto width = extract_int(Data, Size, offset, 1, 32);
        
        // Extract momentum parameter (0.0 to 1.0)
        auto momentum = extract_float(Data, Size, offset, 0.0f, 1.0f);
        
        // Extract exponential_average_factor parameter
        auto exp_avg_factor = extract_float(Data, Size, offset, 0.0f, 1.0f);

        // Create input tensor with random values
        auto input = torch::randn({batch_size, channels, height, width});
        
        // Create running mean and var tensors (should be 1D with size = channels)
        auto running_mean = torch::randn({channels});
        auto running_var = torch::abs(torch::randn({channels})) + 1e-5; // Ensure positive variance
        
        // Test different data types
        std::vector<torch::ScalarType> dtypes = {torch::kFloat32, torch::kFloat64};
        auto dtype_idx = extract_int(Data, Size, offset, 0, dtypes.size() - 1);
        auto dtype = dtypes[dtype_idx];
        
        input = input.to(dtype);
        running_mean = running_mean.to(dtype);
        running_var = running_var.to(dtype);

        // Test batch_norm_update_stats with different parameter combinations
        
        // Test 1: Basic call with momentum
        auto result1 = torch::batch_norm_update_stats(input, running_mean, running_var, momentum);
        
        // Test 2: Call with exponential average factor
        auto result2 = torch::batch_norm_update_stats(input, running_mean, running_var, momentum, exp_avg_factor);
        
        // Test edge cases
        
        // Test 3: Zero momentum
        auto result3 = torch::batch_norm_update_stats(input, running_mean, running_var, 0.0);
        
        // Test 4: Maximum momentum
        auto result4 = torch::batch_norm_update_stats(input, running_mean, running_var, 1.0);
        
        // Test 5: Very small input tensor
        auto small_input = torch::randn({1, 1, 1, 1}).to(dtype);
        auto small_mean = torch::randn({1}).to(dtype);
        auto small_var = torch::abs(torch::randn({1})).to(dtype) + 1e-5;
        auto result5 = torch::batch_norm_update_stats(small_input, small_mean, small_var, momentum);
        
        // Test 6: Different tensor layouts/memory formats
        if (extract_bool(Data, Size, offset)) {
            auto contiguous_input = input.contiguous();
            auto result6 = torch::batch_norm_update_stats(contiguous_input, running_mean, running_var, momentum);
        }
        
        // Test 7: Test with extreme values
        if (extract_bool(Data, Size, offset)) {
            auto extreme_input = input * 1000.0; // Scale up values
            auto result7 = torch::batch_norm_update_stats(extreme_input, running_mean, running_var, momentum);
        }
        
        // Test 8: Test with very small values
        if (extract_bool(Data, Size, offset)) {
            auto tiny_input = input * 1e-6; // Scale down values
            auto result8 = torch::batch_norm_update_stats(tiny_input, running_mean, running_var, momentum);
        }
        
        // Test 9: Test with different input shapes (3D, 5D)
        if (extract_bool(Data, Size, offset)) {
            // 3D input (N, C, L)
            auto input_3d = torch::randn({batch_size, channels, height}).to(dtype);
            auto result9 = torch::batch_norm_update_stats(input_3d, running_mean, running_var, momentum);
            
            // 5D input (N, C, D, H, W)
            auto depth = extract_int(Data, Size, offset, 1, 8);
            auto input_5d = torch::randn({batch_size, channels, depth, height, width}).to(dtype);
            auto result10 = torch::batch_norm_update_stats(input_5d, running_mean, running_var, momentum);
        }
        
        // Test 10: Test with requires_grad tensors
        if (extract_bool(Data, Size, offset)) {
            auto grad_input = input.clone().requires_grad_(true);
            auto grad_mean = running_mean.clone().requires_grad_(true);
            auto grad_var = running_var.clone().requires_grad_(true);
            auto result11 = torch::batch_norm_update_stats(grad_input, grad_mean, grad_var, momentum);
        }
        
        // Test 11: Test with different devices (if CUDA available)
        if (torch::cuda::is_available() && extract_bool(Data, Size, offset)) {
            auto cuda_input = input.to(torch::kCUDA);
            auto cuda_mean = running_mean.to(torch::kCUDA);
            auto cuda_var = running_var.to(torch::kCUDA);
            auto result12 = torch::batch_norm_update_stats(cuda_input, cuda_mean, cuda_var, momentum);
        }
        
        // Verify results are valid tensors
        auto check_tensor = [](const torch::Tensor& t) {
            if (!t.defined()) return false;
            if (torch::any(torch::isnan(t)).item<bool>()) return false;
            if (torch::any(torch::isinf(t)).item<bool>()) return false;
            return true;
        };
        
        // Basic validation of results
        if (!check_tensor(std::get<0>(result1)) || !check_tensor(std::get<1>(result1))) {
            return -1;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}