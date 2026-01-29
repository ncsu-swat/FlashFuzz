#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <cstring>

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
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        bool affine = (offset < Size) ? (Data[offset++] & 0x1) : false;
        bool track_running_stats = (offset < Size) ? (Data[offset++] & 0x1) : false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            eps = std::abs(eps);
            if (eps < 1e-10 || !std::isfinite(eps)) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            momentum = std::abs(momentum);
            if (!std::isfinite(momentum) || momentum > 1.0) {
                momentum = 0.1;
            }
        }
        
        int64_t ndim = input.dim();
        
        // InstanceNorm requires at least 3D tensor for 1D, 4D for 2D, 5D for 3D
        // Format: (N, C, ...) where ... is the spatial dimensions
        if (ndim < 3) {
            return 0;
        }
        
        int64_t num_features = input.size(1);
        if (num_features <= 0) {
            return 0;
        }
        
        // Ensure input is float type for normalization
        torch::Tensor float_input = input.to(torch::kFloat);
        
        try {
            if (ndim == 3) {
                // InstanceNorm1d: (N, C, L)
                torch::nn::InstanceNorm1d instance_norm(
                    torch::nn::InstanceNorm1dOptions(num_features)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats));
                
                auto output = instance_norm(float_input);
                
                // Test eval mode
                instance_norm->eval();
                auto output_eval = instance_norm(float_input);
            }
            else if (ndim == 4) {
                // InstanceNorm2d: (N, C, H, W)
                torch::nn::InstanceNorm2d instance_norm(
                    torch::nn::InstanceNorm2dOptions(num_features)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats));
                
                auto output = instance_norm(float_input);
                
                // Test eval mode
                instance_norm->eval();
                auto output_eval = instance_norm(float_input);
            }
            else if (ndim == 5) {
                // InstanceNorm3d: (N, C, D, H, W)
                torch::nn::InstanceNorm3d instance_norm(
                    torch::nn::InstanceNorm3dOptions(num_features)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats));
                
                auto output = instance_norm(float_input);
                
                // Test eval mode
                instance_norm->eval();
                auto output_eval = instance_norm(float_input);
            }
        } catch (const c10::Error&) {
            // Expected failures for invalid shapes/sizes - silently continue
        }
        
        // Additional test: try with double precision
        if (offset < Size) {
            try {
                torch::Tensor double_input = input.to(torch::kDouble);
                
                if (ndim == 4) {
                    torch::nn::InstanceNorm2d instance_norm(
                        torch::nn::InstanceNorm2dOptions(num_features)
                            .eps(eps)
                            .momentum(momentum)
                            .affine(affine)
                            .track_running_stats(track_running_stats));
                    instance_norm->to(torch::kDouble);
                    auto output = instance_norm(double_input);
                }
            } catch (const c10::Error&) {
                // Expected failures - silently continue
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