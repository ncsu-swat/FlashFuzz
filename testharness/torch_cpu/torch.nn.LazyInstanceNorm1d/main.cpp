#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters before creating tensor
        bool affine = Data[offset++] % 2 == 1;
        bool track_running_stats = Data[offset++] % 2 == 1;
        
        // Extract epsilon (small value to prevent division by zero)
        double eps = 1e-5;
        if (offset + 2 <= Size) {
            uint16_t eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(eps_raw));
            offset += sizeof(eps_raw);
            // Convert to a small positive value
            eps = 1e-10 + (eps_raw % 1000) * 1e-6;
        }
        
        // Extract momentum
        double momentum = 0.1;
        if (offset + 2 <= Size) {
            uint16_t momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(momentum_raw));
            offset += sizeof(momentum_raw);
            // Convert to a value between 0 and 1
            momentum = static_cast<double>(momentum_raw % 1000) / 1000.0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is float type for normalization
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has correct shape for InstanceNorm1d
        // InstanceNorm1d expects input of shape (N, C, L) or (C, L)
        // where C is the number of features (channels)
        if (input.dim() == 0) {
            // Scalar - reshape to (1, 1, 1)
            input = input.reshape({1, 1, 1});
        } else if (input.dim() == 1) {
            // If 1D with size L, reshape to (1, 1, L) treating as single channel
            input = input.reshape({1, 1, input.size(0)});
        } else if (input.dim() == 2) {
            // If 2D (C, L), add batch dimension to get (1, C, L)
            input = input.unsqueeze(0);
        } else if (input.dim() > 3) {
            // If more than 3D, reshape to 3D
            int64_t n = input.size(0);
            int64_t c = input.size(1);
            int64_t l = input.numel() / (n * c);
            if (l < 1) l = 1;
            try {
                input = input.reshape({n, c, l});
            } catch (...) {
                // If reshape fails, create a simple valid tensor
                input = input.flatten().slice(0, 0, std::min(input.numel(), (int64_t)64)).reshape({1, 1, -1});
            }
        }
        
        // Ensure we have at least some length dimension
        if (input.size(2) == 0) {
            return 0;
        }
        
        // Get num_features from the input tensor (channel dimension)
        int64_t num_features = input.size(1);
        
        // Create InstanceNorm1d module with the inferred num_features
        // Note: LazyInstanceNorm1d is Python-only, so we use InstanceNorm1d in C++
        auto options = torch::nn::InstanceNorm1dOptions(num_features)
            .eps(eps)
            .momentum(momentum)
            .affine(affine)
            .track_running_stats(track_running_stats);
        
        torch::nn::InstanceNorm1d instance_norm(options);
        
        // Apply the operation
        torch::Tensor output;
        try {
            output = instance_norm(input);
        } catch (const c10::Error &e) {
            // Shape/dimension errors are expected with fuzzed input
            return 0;
        }
        
        // Force computation to ensure any errors are caught
        output = output.contiguous();
        
        // Access some elements to ensure computation is performed
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test with a second input of different batch size but same channels
        // This exercises the module with different input shapes
        if (input.size(0) > 0 && input.size(1) > 0) {
            int64_t new_batch = (input.size(0) % 3) + 1;
            int64_t new_length = (input.size(2) % 5) + 1;
            torch::Tensor input2 = torch::randn({new_batch, num_features, new_length});
            try {
                torch::Tensor output2 = instance_norm(input2);
                output2 = output2.contiguous();
            } catch (const c10::Error &e) {
                // Expected for some configurations
            }
        }
        
        // Test eval mode as well
        instance_norm->eval();
        try {
            torch::Tensor output3 = instance_norm(input);
            output3 = output3.contiguous();
        } catch (const c10::Error &e) {
            // Expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}