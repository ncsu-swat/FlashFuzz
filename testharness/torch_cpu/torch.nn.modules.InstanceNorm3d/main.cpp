#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters for InstanceNorm3d from the input data first
        bool affine = (Data[offset++] % 2 == 0);
        bool track_running_stats = (Data[offset++] % 2 == 0);
        
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps = std::abs(static_cast<double>(eps_raw));
            if (eps < 1e-10 || std::isnan(eps) || std::isinf(eps)) {
                eps = 1e-5;
            }
            if (eps > 1.0) {
                eps = 1e-5;
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            momentum = std::abs(static_cast<double>(momentum_raw));
            if (std::isnan(momentum) || std::isinf(momentum)) {
                momentum = 0.1;
            }
            momentum = std::fmod(momentum, 1.0);
        }
        
        // Create input tensor - must be 5D for InstanceNorm3d (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is float type for normalization
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure tensor has 5 dimensions for InstanceNorm3d
        if (input.dim() == 0) {
            input = input.reshape({1, 1, 1, 1, 1});
        } else if (input.dim() == 1) {
            input = input.reshape({1, 1, 1, 1, input.size(0)});
        } else if (input.dim() == 2) {
            input = input.reshape({1, 1, 1, input.size(0), input.size(1)});
        } else if (input.dim() == 3) {
            input = input.reshape({1, 1, input.size(0), input.size(1), input.size(2)});
        } else if (input.dim() == 4) {
            input = input.reshape({1, input.size(0), input.size(1), input.size(2), input.size(3)});
        } else if (input.dim() > 5) {
            // Flatten to 5D
            int64_t total = input.numel();
            input = input.reshape({1, 1, 1, 1, total});
        }
        
        // Get the number of channels (2nd dimension)
        int64_t num_features = input.size(1);
        if (num_features <= 0) {
            num_features = 1;
            input = input.reshape({input.size(0), 1, input.size(2), input.size(3), input.size(4)});
        }
        
        // Limit num_features to avoid excessive memory usage
        if (num_features > 1024) {
            return 0;
        }
        
        // Create InstanceNorm3d module
        torch::nn::InstanceNorm3d instance_norm(
            torch::nn::InstanceNorm3dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats));
        
        // Apply InstanceNorm3d in train mode (default)
        torch::Tensor output = instance_norm->forward(input);
        
        // Access output properties to ensure computation completed
        auto output_size = output.sizes();
        (void)output_size;
        
        // Test eval mode
        instance_norm->eval();
        torch::Tensor output_eval = instance_norm->forward(input);
        (void)output_eval;
        
        // Test train mode again
        instance_norm->train();
        torch::Tensor output_train = instance_norm->forward(input);
        (void)output_train;
        
        // Test with different input types if possible
        if (offset < Size) {
            auto dtype_selector = Data[offset++];
            
            try {
                torch::Dtype dtype;
                switch (dtype_selector % 3) {
                    case 0:
                        dtype = torch::kFloat32;
                        break;
                    case 1:
                        dtype = torch::kFloat64;
                        break;
                    case 2:
                        dtype = torch::kFloat16;
                        break;
                    default:
                        dtype = torch::kFloat32;
                }
                
                auto input_converted = input.to(dtype);
                // Need to recreate the module for different dtypes
                torch::nn::InstanceNorm3d instance_norm_converted(
                    torch::nn::InstanceNorm3dOptions(num_features)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats));
                instance_norm_converted->to(dtype);
                auto output_converted = instance_norm_converted->forward(input_converted);
                (void)output_converted;
            } catch (const std::exception&) {
                // Some dtype conversions might not be valid, ignore silently
            }
        }
        
        // Test with batch size > 1 if we have enough data
        if (input.size(0) == 1 && input.numel() >= 2) {
            try {
                auto batched_input = torch::cat({input, input}, 0);
                auto batched_output = instance_norm->forward(batched_input);
                (void)batched_output;
            } catch (const std::exception&) {
                // Ignore batch concatenation failures
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