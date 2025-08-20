#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for InstanceNorm
        bool affine = (offset < Size) ? (Data[offset++] & 0x1) : false;
        bool track_running_stats = (offset < Size) ? (Data[offset++] & 0x1) : false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Get the number of dimensions
        int64_t ndim = input.dim();
        
        // InstanceNorm requires at least 2D tensor (N,C,...)
        if (ndim >= 2) {
            // Get the number of features (channels)
            int64_t num_features = input.size(1);
            
            // Create appropriate InstanceNorm module based on dimensions
            if (ndim == 2) {
                torch::nn::InstanceNorm1d instance_norm(
                    torch::nn::InstanceNorm1dOptions(num_features)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats));
                
                auto output = instance_norm(input);
            }
            else if (ndim == 3) {
                torch::nn::InstanceNorm1d instance_norm(
                    torch::nn::InstanceNorm1dOptions(num_features)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats));
                
                auto output = instance_norm(input);
            }
            else if (ndim == 4) {
                torch::nn::InstanceNorm2d instance_norm(
                    torch::nn::InstanceNorm2dOptions(num_features)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats));
                
                auto output = instance_norm(input);
            }
            else if (ndim == 5) {
                torch::nn::InstanceNorm3d instance_norm(
                    torch::nn::InstanceNorm3dOptions(num_features)
                        .eps(eps)
                        .momentum(momentum)
                        .affine(affine)
                        .track_running_stats(track_running_stats));
                
                auto output = instance_norm(input);
            }
        }
        
        // Try with different data types
        if (offset < Size && ndim >= 2) {
            int64_t num_features = input.size(1);
            
            // Convert to float if not already
            torch::Tensor float_input = input.to(torch::kFloat);
            
            torch::nn::InstanceNorm2d instance_norm(
                torch::nn::InstanceNorm2dOptions(num_features)
                    .eps(eps)
                    .momentum(momentum)
                    .affine(affine)
                    .track_running_stats(track_running_stats));
            
            auto output = instance_norm(float_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}