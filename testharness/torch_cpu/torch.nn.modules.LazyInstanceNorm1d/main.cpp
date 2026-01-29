#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for LazyInstanceNorm1d from data
        bool affine = Data[offset++] & 1;
        bool track_running_stats = Data[offset++] & 1;
        
        // Extract eps (use bytes to create a reasonable epsilon)
        uint8_t eps_byte = Data[offset++];
        double eps = 1e-7 + (eps_byte / 255.0) * 1e-3;  // Range: 1e-7 to ~1e-3
        
        // Extract momentum
        uint8_t momentum_byte = Data[offset++];
        double momentum = 0.01 + (momentum_byte / 255.0) * 0.5;  // Range: 0.01 to 0.51
        
        // Create input tensor - InstanceNorm1d expects 3D: (N, C, L)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is 3D for InstanceNorm1d (batch, channels, length)
        if (input.numel() == 0) {
            return 0;
        }
        
        // Reshape input to 3D if needed
        if (input.dim() == 0) {
            input = input.reshape({1, 1, 1});
        } else if (input.dim() == 1) {
            int64_t len = input.size(0);
            if (len >= 2) {
                input = input.reshape({1, 1, len});
            } else {
                input = input.reshape({1, 1, 1});
            }
        } else if (input.dim() == 2) {
            // (N, C) -> (N, C, 1) or treat as (1, C, L)
            int64_t d0 = input.size(0);
            int64_t d1 = input.size(1);
            if (d0 > 0 && d1 > 0) {
                input = input.reshape({1, d0, d1});
            } else {
                return 0;
            }
        } else if (input.dim() > 3) {
            // Flatten extra dimensions into length dimension
            int64_t batch = input.size(0);
            int64_t channels = input.size(1);
            int64_t length = input.numel() / (batch * channels);
            if (batch <= 0 || channels <= 0 || length <= 0) {
                return 0;
            }
            input = input.reshape({batch, channels, length});
        }
        
        // Ensure we have valid dimensions
        if (input.size(0) <= 0 || input.size(1) <= 0 || input.size(2) <= 0) {
            return 0;
        }
        
        // Convert to float for normalization
        input = input.to(torch::kFloat32);
        
        // Create LazyInstanceNorm1d module - it infers num_features from first input
        auto lazy_instance_norm = torch::nn::LazyInstanceNorm1d(
            torch::nn::LazyInstanceNorm1dOptions()
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the module to the input tensor (this materializes the lazy module)
        torch::Tensor output;
        try {
            output = lazy_instance_norm->forward(input);
        } catch (const c10::Error&) {
            // Shape/type errors are expected with fuzzed inputs
            return 0;
        }
        
        // Basic output validation (silent, don't log expected conditions)
        if (output.numel() == 0) {
            return 0;
        }
        
        // Test the module in eval mode
        lazy_instance_norm->eval();
        try {
            torch::Tensor eval_output = lazy_instance_norm->forward(input);
        } catch (const c10::Error&) {
            // Expected for some edge cases
        }
        
        // Switch back to train mode and test again
        lazy_instance_norm->train();
        try {
            torch::Tensor train_output = lazy_instance_norm->forward(input);
        } catch (const c10::Error&) {
            // Expected for some edge cases
        }
        
        // Test with different batch sizes if possible
        if (input.size(0) > 1) {
            try {
                torch::Tensor half_input = input.slice(0, 0, input.size(0) / 2 + 1);
                torch::Tensor half_output = lazy_instance_norm->forward(half_input);
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }
        
        // Test with a new LazyInstanceNorm1d to exercise lazy initialization again
        if (offset + 4 < Size) {
            auto lazy_instance_norm2 = torch::nn::LazyInstanceNorm1d(
                torch::nn::LazyInstanceNorm1dOptions()
                    .eps(eps)
                    .affine(!affine)  // Try opposite affine setting
                    .track_running_stats(!track_running_stats)
            );
            
            try {
                torch::Tensor output2 = lazy_instance_norm2->forward(input);
            } catch (const c10::Error&) {
                // Expected
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