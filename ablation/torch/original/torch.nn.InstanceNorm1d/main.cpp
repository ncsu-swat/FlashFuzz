#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 10) {
        // Need minimum bytes for module params and tensor creation
        return 0;
    }

    try
    {
        size_t offset = 0;
        
        // Parse InstanceNorm1d parameters from fuzzer input
        // 1. num_features (required parameter)
        int64_t num_features = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&num_features, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Constrain to reasonable range [1, 2048]
            num_features = 1 + (std::abs(num_features) % 2048);
        }
        
        // 2. eps (epsilon for numerical stability)
        double eps = 1e-5;
        if (offset + sizeof(double) <= size) {
            std::memcpy(&eps, data + offset, sizeof(double));
            offset += sizeof(double);
            // Constrain to reasonable range [1e-10, 1.0]
            eps = std::abs(eps);
            if (eps < 1e-10) eps = 1e-10;
            if (eps > 1.0) eps = 1.0;
        }
        
        // 3. momentum (for running stats)
        double momentum = 0.1;
        if (offset + sizeof(double) <= size) {
            std::memcpy(&momentum, data + offset, sizeof(double));
            offset += sizeof(double);
            // Constrain to [0.0, 1.0]
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = std::fmod(momentum, 1.0);
        }
        
        // 4. affine (boolean)
        bool affine = false;
        if (offset < size) {
            affine = (data[offset++] % 2) == 1;
        }
        
        // 5. track_running_stats (boolean)
        bool track_running_stats = false;
        if (offset < size) {
            track_running_stats = (data[offset++] % 2) == 1;
        }
        
        // 6. training mode (boolean)
        bool training_mode = true;
        if (offset < size) {
            training_mode = (data[offset++] % 2) == 1;
        }
        
        // Create InstanceNorm1d module with parsed parameters
        auto options = torch::nn::InstanceNorm1dOptions(num_features)
            .eps(eps)
            .momentum(momentum)
            .affine(affine)
            .track_running_stats(track_running_stats);
        
        torch::nn::InstanceNorm1d instance_norm(options);
        
        // Set training/eval mode
        if (training_mode) {
            instance_norm->train();
        } else {
            instance_norm->eval();
        }
        
        // Parse input tensor from remaining bytes
        if (offset >= size) {
            // No data left for tensor, create a default one
            auto input = torch::randn({2, static_cast<int64_t>(num_features), 10});
            auto output = instance_norm->forward(input);
            return 0;
        }
        
        // Create input tensor using fuzzer_utils
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(data, size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, create a valid default tensor
            input = torch::randn({2, static_cast<int64_t>(num_features), 10});
        }
        
        // Validate and potentially reshape input tensor for InstanceNorm1d
        // InstanceNorm1d expects either (N, C, L) or (C, L)
        auto input_dim = input.dim();
        
        if (input_dim == 0) {
            // Scalar - reshape to valid format
            input = input.reshape({1, 1, 1});
        } else if (input_dim == 1) {
            // 1D tensor - treat as (C=1, L=size)
            input = input.reshape({1, input.size(0)});
        } else if (input_dim == 2) {
            // 2D tensor (C, L) - valid as is
            // Ensure C matches num_features or adjust
            if (input.size(0) != num_features) {
                // Either resize or create new tensor
                int64_t L = input.size(1);
                if (L == 0) L = 1;
                input = torch::randn({static_cast<int64_t>(num_features), L}, input.options());
            }
        } else if (input_dim == 3) {
            // 3D tensor (N, C, L) - valid, check C dimension
            if (input.size(1) != num_features) {
                int64_t N = input.size(0);
                int64_t L = input.size(2);
                if (N == 0) N = 1;
                if (L == 0) L = 1;
                input = torch::randn({N, static_cast<int64_t>(num_features), L}, input.options());
            }
        } else {
            // Higher dimensional - reshape to 3D
            int64_t total_elements = input.numel();
            if (total_elements == 0) {
                input = torch::randn({1, static_cast<int64_t>(num_features), 1});
            } else {
                int64_t L = std::max(int64_t(1), total_elements / num_features);
                if (L > 10000) L = 10000; // Cap sequence length
                input = input.flatten().narrow(0, 0, std::min(input.numel(), num_features * L));
                input = input.reshape({1, static_cast<int64_t>(num_features), L});
            }
        }
        
        // Test forward pass
        torch::Tensor output;
        output = instance_norm->forward(input);
        
        // Additional operations to increase coverage
        
        // Test with different input variations
        if (offset < size && (data[offset] % 3) == 0) {
            // Test with zeros
            auto zero_input = torch::zeros_like(input);
            auto zero_output = instance_norm->forward(zero_input);
        }
        
        if (offset < size && (data[offset] % 3) == 1) {
            // Test with ones
            auto ones_input = torch::ones_like(input);
            auto ones_output = instance_norm->forward(ones_input);
        }
        
        if (offset < size && (data[offset] % 3) == 2) {
            // Test with very large values
            auto large_input = input * 1e6;
            auto large_output = instance_norm->forward(large_input);
        }
        
        // Test gradient computation if in training mode and affine is true
        if (training_mode && affine && input.requires_grad()) {
            try {
                auto loss = output.sum();
                loss.backward();
            } catch (...) {
                // Ignore gradient computation errors
            }
        }
        
        // Test state dict operations
        try {
            auto state_dict = instance_norm->named_parameters();
            
            // If affine is true, we should have weight and bias parameters
            if (affine) {
                for (const auto& pair : state_dict) {
                    auto param = pair.value();
                    // Access parameter to ensure it's valid
                    auto param_sum = param.sum();
                }
            }
            
            // Test running stats if tracked
            if (track_running_stats) {
                auto buffers = instance_norm->named_buffers();
                for (const auto& pair : buffers) {
                    auto buffer = pair.value();
                    // Access buffer to ensure it's valid
                    if (buffer.defined()) {
                        auto buffer_sum = buffer.sum();
                    }
                }
            }
        } catch (...) {
            // Ignore state dict errors
        }
        
        // Test with batch of different sizes
        if (input.dim() == 3 && offset < size) {
            uint8_t new_batch_size = data[offset] % 16 + 1;
            try {
                auto new_input = torch::randn({static_cast<int64_t>(new_batch_size), 
                                              static_cast<int64_t>(num_features), 
                                              input.size(2)}, input.options());
                auto new_output = instance_norm->forward(new_input);
            } catch (...) {
                // Ignore errors from batch size changes
            }
        }
        
        // Test edge cases with special float values if dtype is floating point
        if (input.is_floating_point()) {
            try {
                // Test with NaN
                auto nan_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                auto nan_output = instance_norm->forward(nan_input);
                
                // Test with Inf
                auto inf_input = torch::full_like(input, std::numeric_limits<float>::infinity());
                auto inf_output = instance_norm->forward(inf_input);
                
                // Test with -Inf
                auto ninf_input = torch::full_like(input, -std::numeric_limits<float>::infinity());
                auto ninf_output = instance_norm->forward(ninf_input);
            } catch (...) {
                // Ignore errors from special values
            }
        }
        
        // Test module cloning
        try {
            auto cloned_module = instance_norm->clone();
            auto cloned_output = cloned_module->forward(input);
            
            // Compare outputs if both successful
            if (output.defined() && cloned_output.defined()) {
                bool outputs_equal = torch::allclose(output, cloned_output, 1e-5, 1e-8);
            }
        } catch (...) {
            // Ignore cloning errors
        }
        
        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        // Catch any other exceptions
        return -1;
    }
}