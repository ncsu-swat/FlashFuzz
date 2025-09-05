#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 10) {
            // Need minimum bytes for basic parsing
            return 0;
        }

        size_t offset = 0;

        // Parse configuration bytes for InstanceNorm1d
        uint8_t num_features_byte = Data[offset++];
        uint8_t eps_byte = Data[offset++];
        uint8_t momentum_byte = Data[offset++];
        uint8_t affine_byte = Data[offset++];
        uint8_t track_running_stats_byte = Data[offset++];
        uint8_t training_mode_byte = Data[offset++];

        // Convert bytes to parameters
        // num_features: 1 to 256 channels
        int64_t num_features = 1 + (num_features_byte % 256);
        
        // eps: small positive value for numerical stability
        double eps = 1e-8 + (eps_byte / 255.0) * 1e-3;
        
        // momentum: 0.0 to 1.0
        double momentum = (momentum_byte / 255.0);
        
        // Boolean flags
        bool affine = affine_byte & 0x01;
        bool track_running_stats = track_running_stats_byte & 0x01;
        bool training_mode = training_mode_byte & 0x01;

        // Create InstanceNorm1d module
        torch::nn::InstanceNorm1dOptions options(num_features);
        options.eps(eps);
        options.momentum(momentum);
        options.affine(affine);
        options.track_running_stats(track_running_stats);
        
        torch::nn::InstanceNorm1d norm_module(options);
        
        // Set training/eval mode
        if (training_mode) {
            norm_module->train();
        } else {
            norm_module->eval();
        }

        // Create input tensor from fuzzer data
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with minimal valid tensor
            input = torch::randn({2, num_features, 4});
        }

        // Ensure input has correct number of dimensions for InstanceNorm1d
        // InstanceNorm1d expects 2D (N, C) or 3D (N, C, L) input
        if (input.dim() < 2) {
            // Reshape to have at least 2 dimensions
            int64_t total_elements = input.numel();
            if (total_elements == 0) {
                input = torch::randn({1, num_features});
            } else if (total_elements < num_features) {
                // Pad with zeros if needed
                input = torch::randn({1, num_features});
            } else {
                // Reshape to [batch_size, num_features, ...]
                int64_t batch_size = std::max(int64_t(1), total_elements / num_features);
                input = input.view({batch_size, num_features});
            }
        } else if (input.dim() > 3) {
            // Flatten extra dimensions
            auto sizes = input.sizes();
            int64_t batch_size = sizes[0];
            int64_t channels = std::min(sizes[1], num_features);
            int64_t length = 1;
            for (int i = 2; i < input.dim(); ++i) {
                length *= sizes[i];
            }
            input = input.view({batch_size, channels, length});
        }

        // Adjust channel dimension if needed
        if (input.size(1) != num_features) {
            auto sizes = input.sizes().vec();
            if (input.size(1) < num_features) {
                // Pad channels with zeros
                std::vector<torch::Tensor> padding;
                padding.push_back(input);
                int64_t pad_channels = num_features - input.size(1);
                sizes[1] = pad_channels;
                padding.push_back(torch::zeros(sizes, input.options()));
                input = torch::cat(padding, 1);
            } else {
                // Truncate channels
                input = input.narrow(1, 0, num_features);
            }
        }

        // Ensure input is floating point (InstanceNorm requires float types)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }

        // Apply InstanceNorm1d
        torch::Tensor output;
        try {
            output = norm_module->forward(input);
            
            // Perform additional operations to increase coverage
            
            // Test with different input shapes while maintaining num_features
            if (offset < Size - 2) {
                uint8_t reshape_type = Data[offset++];
                switch (reshape_type % 4) {
                    case 0:
                        // 2D input (N, C)
                        if (input.dim() == 3 && input.size(2) > 1) {
                            auto flat_input = input.view({input.size(0) * input.size(2), num_features});
                            auto flat_output = norm_module->forward(flat_input);
                        }
                        break;
                    case 1:
                        // 3D input with different length
                        if (input.dim() == 2) {
                            auto expanded_input = input.unsqueeze(2).expand({input.size(0), num_features, 8});
                            auto expanded_output = norm_module->forward(expanded_input);
                        }
                        break;
                    case 2:
                        // Test with batch size 1
                        if (input.size(0) > 1) {
                            auto single_batch = input.narrow(0, 0, 1);
                            auto single_output = norm_module->forward(single_batch);
                        }
                        break;
                    case 3:
                        // Test with contiguous and non-contiguous tensors
                        if (input.is_contiguous()) {
                            auto transposed = input.transpose(0, 1).transpose(0, 1);
                            auto trans_output = norm_module->forward(transposed);
                        }
                        break;
                }
            }
            
            // Test gradient computation if in training mode
            if (training_mode && input.requires_grad()) {
                try {
                    auto loss = output.mean();
                    loss.backward();
                } catch (...) {
                    // Ignore gradient computation errors
                }
            }
            
            // Access and modify module parameters if affine is true
            if (affine) {
                auto params = norm_module->named_parameters();
                for (const auto& param : params) {
                    auto tensor = param.value();
                    // Trigger parameter access
                    auto sum = tensor.sum();
                }
            }
            
            // Access running stats if tracking
            if (track_running_stats) {
                auto buffers = norm_module->named_buffers();
                for (const auto& buffer : buffers) {
                    auto tensor = buffer.value();
                    // Trigger buffer access
                    auto mean_val = tensor.mean();
                }
            }
            
            // Test state dict operations
            try {
                auto state = norm_module->state_dict();
                torch::nn::InstanceNorm1d new_module(options);
                new_module->load_state_dict(state);
                auto new_output = new_module->forward(input);
            } catch (...) {
                // Ignore state dict errors
            }

        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for edge cases
            return 0;
        } catch (const std::exception& e) {
            // Other exceptions might indicate issues
            return 0;
        }

        // Test edge cases with special values
        if (offset < Size - 1) {
            uint8_t special_case = Data[offset++];
            torch::Tensor special_input;
            
            switch (special_case % 6) {
                case 0:
                    // Test with zeros
                    special_input = torch::zeros_like(input);
                    break;
                case 1:
                    // Test with ones
                    special_input = torch::ones_like(input);
                    break;
                case 2:
                    // Test with very large values
                    special_input = input * 1e10;
                    break;
                case 3:
                    // Test with very small values
                    special_input = input * 1e-10;
                    break;
                case 4:
                    // Test with NaN
                    special_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                    break;
                case 5:
                    // Test with Inf
                    special_input = torch::full_like(input, std::numeric_limits<float>::infinity());
                    break;
            }
            
            try {
                auto special_output = norm_module->forward(special_input);
            } catch (...) {
                // Ignore errors from special values
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}