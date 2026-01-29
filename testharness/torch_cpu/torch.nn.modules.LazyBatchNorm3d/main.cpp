#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip if we don't have enough data
        if (Size < 10) {
            return 0;
        }
        
        // Determine dimensions from fuzzer data
        int64_t batch_size = (Data[offset++] % 8) + 1;
        int64_t num_channels = (Data[offset++] % 64) + 1;
        int64_t depth = (Data[offset++] % 8) + 1;
        int64_t height = (Data[offset++] % 8) + 1;
        int64_t width = (Data[offset++] % 8) + 1;
        
        // Create input tensor with 5 dimensions (N, C, D, H, W) for BatchNorm3d
        std::vector<int64_t> input_shape = {batch_size, num_channels, depth, height, width};
        torch::Tensor input = torch::randn(input_shape);
        
        // Create BatchNorm3d module options
        // Note: LazyBatchNorm3d is not available in C++ API, using BatchNorm3d instead
        auto bn_options = torch::nn::BatchNorm3dOptions(num_channels);
        
        // Configure module parameters based on remaining data
        if (offset + 3 < Size) {
            // Set eps (small value added to variance for numerical stability)
            double eps = static_cast<double>(Data[offset++]) / 255.0 * 0.1 + 1e-5;
            bn_options.eps(eps);
            
            // Set momentum
            double momentum = static_cast<double>(Data[offset++]) / 255.0;
            bn_options.momentum(momentum);
            
            // Set affine flag
            bool affine = Data[offset++] % 2 == 0;
            bn_options.affine(affine);
            
            // Set track_running_stats flag
            bool track_running_stats = Data[offset++] % 2 == 0;
            bn_options.track_running_stats(track_running_stats);
        }
        
        torch::nn::BatchNorm3d bn(bn_options);
        
        // Apply the BatchNorm3d to the input tensor
        torch::Tensor output = bn->forward(input);
        
        // Verify output shape matches input shape
        if (output.sizes() != input.sizes()) {
            std::cerr << "Shape mismatch after forward" << std::endl;
        }
        
        // Test the module in training mode
        bn->train();
        torch::Tensor output_train = bn->forward(input);
        
        // Test in evaluation mode
        bn->eval();
        torch::Tensor output_eval = bn->forward(input);
        
        // Test with a second input of same channel dimension but different spatial dims
        if (offset + 3 < Size) {
            std::vector<int64_t> new_shape2;
            new_shape2.push_back((Data[offset++] % 4) + 1);  // batch
            new_shape2.push_back(num_channels);  // same channels
            new_shape2.push_back((Data[offset++] % 8) + 1);  // D
            new_shape2.push_back((Data[offset++] % 8) + 1);  // H
            new_shape2.push_back((Data[offset++] % 8) + 1);  // W
            
            torch::Tensor input2 = torch::randn(new_shape2);
            torch::Tensor output2 = bn->forward(input2);
        }
        
        // Test double precision
        try {
            auto input_double = torch::randn(input_shape, torch::kFloat64);
            torch::nn::BatchNorm3d bn_double(torch::nn::BatchNorm3dOptions(num_channels));
            auto output_double = bn_double->forward(input_double);
        } catch (...) {
            // Ignore errors with double precision
        }
        
        // Test half precision if available
        try {
            auto input_half = torch::randn(input_shape, torch::kFloat16);
            torch::nn::BatchNorm3d bn_half(torch::nn::BatchNorm3dOptions(num_channels));
            auto output_half = bn_half->forward(input_half);
        } catch (...) {
            // Ignore errors with half precision
        }
        
        // Test with different input using createTensor from fuzzer
        torch::Tensor fuzz_input = fuzzer_utils::createTensor(Data, Size, offset);
        if (fuzz_input.numel() > 0) {
            // Reshape to 5D with matching channels
            int64_t total = fuzz_input.numel();
            int64_t spatial = std::max<int64_t>(1, total / num_channels);
            int64_t side = std::max<int64_t>(1, static_cast<int64_t>(std::cbrt(spatial)));
            
            try {
                auto reshaped = fuzz_input.reshape({1, num_channels, side, side, side});
                if (!reshaped.is_floating_point()) {
                    reshaped = reshaped.to(torch::kFloat32);
                }
                auto fuzz_output = bn->forward(reshaped);
            } catch (...) {
                // Shape may not work out, ignore
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