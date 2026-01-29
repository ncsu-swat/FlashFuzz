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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for InstanceNorm1d from the remaining data
        bool affine = false;
        bool track_running_stats = false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 2 <= Size) {
            affine = Data[offset++] & 0x1;
            track_running_stats = Data[offset++] & 0x1;
        }
        
        if (offset + sizeof(double) <= Size) {
            eps = *reinterpret_cast<const double*>(Data + offset);
            offset += sizeof(double);
            // Ensure eps is positive and reasonable
            eps = std::abs(eps);
            if (eps == 0 || std::isnan(eps) || std::isinf(eps)) eps = 1e-5;
            if (eps < 1e-10) eps = 1e-10;
            if (eps > 1.0) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            momentum = *reinterpret_cast<const double*>(Data + offset);
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (std::isnan(momentum) || std::isinf(momentum)) momentum = 0.1;
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // InstanceNorm1d expects input of shape [N, C, L] (3D)
        // Reshape input to 3D if necessary
        if (input.dim() == 0) {
            // Scalar -> [1, 1, 1]
            input = input.reshape({1, 1, 1});
        } else if (input.dim() == 1) {
            // [L] -> [1, 1, L]
            input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 2) {
            // [N, L] -> [N, 1, L]
            input = input.unsqueeze(1);
        } else if (input.dim() > 3) {
            // Flatten extra dimensions into the last dimension
            auto sizes = input.sizes();
            int64_t N = sizes[0];
            int64_t C = sizes[1];
            int64_t L = 1;
            for (int64_t i = 2; i < input.dim(); i++) {
                L *= sizes[i];
            }
            input = input.reshape({N, C, L});
        }
        // Now input is guaranteed to be 3D: [N, C, L]
        
        // Get the number of features (channels) from the input tensor
        int64_t num_features = input.size(1);
        
        // Ensure num_features is at least 1
        if (num_features < 1) {
            return 0;
        }
        
        // Ensure input is floating point for normalization
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create InstanceNorm1d module
        torch::nn::InstanceNorm1d instance_norm(
            torch::nn::InstanceNorm1dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply InstanceNorm1d to the input tensor
        torch::Tensor output = instance_norm(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        (void)sum;
        
        // Test eval mode vs train mode
        if (offset < Size && (Data[offset++] & 0x1)) {
            instance_norm->eval();
            try {
                output = instance_norm(input);
            } catch (const std::exception&) {
                // Eval mode might behave differently
            }
        }
        
        // Test with different sequence lengths
        if (offset + 1 < Size && input.size(2) > 1) {
            uint8_t len_mod = Data[offset++];
            int64_t new_len = 1 + (len_mod % input.size(2));
            
            try {
                torch::Tensor sliced = input.slice(2, 0, new_len);
                output = instance_norm(sliced);
            } catch (const std::exception&) {
                // Slicing might cause issues
            }
        }
        
        // Test with different batch sizes via slicing
        if (offset < Size && input.size(0) > 1) {
            uint8_t batch_mod = Data[offset++];
            int64_t new_batch = 1 + (batch_mod % input.size(0));
            
            try {
                torch::Tensor sliced = input.slice(0, 0, new_batch);
                output = instance_norm(sliced);
            } catch (const std::exception&) {
                // Batch slicing might cause issues
            }
        }
        
        // Test with double precision
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                torch::Tensor double_input = input.to(torch::kFloat64);
                
                torch::nn::InstanceNorm1d instance_norm_double(
                    torch::nn::InstanceNorm1dOptions(num_features)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats)
                );
                instance_norm_double->to(torch::kFloat64);
                
                output = instance_norm_double(double_input);
            } catch (const std::exception&) {
                // Double conversion might fail
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