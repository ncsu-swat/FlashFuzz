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
        
        // Ensure input has at least 4 dimensions for InstanceNorm2d (N, C, H, W)
        if (input.dim() < 4) {
            // Expand dimensions if needed
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for InstanceNorm2d from the remaining data
        bool affine = false;
        bool track_running_stats = false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 2 <= Size) {
            affine = Data[offset++] & 0x1;
            track_running_stats = Data[offset++] & 0x1;
        }
        
        if (offset + sizeof(double) <= Size) {
            memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive and not too small
            eps = std::abs(eps);
            if (eps < 1e-10) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is in [0, 1]
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Get the number of channels (second dimension)
        int64_t num_features = input.size(1);
        
        // Create InstanceNorm2d module
        torch::nn::InstanceNorm2d instance_norm(
            torch::nn::InstanceNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply InstanceNorm2d
        torch::Tensor output = instance_norm->forward(input);
        
        // Try to access some properties to ensure they're properly initialized
        if (affine) {
            torch::Tensor weight = instance_norm->weight;
            torch::Tensor bias = instance_norm->bias;
        }
        
        if (track_running_stats) {
            torch::Tensor running_mean = instance_norm->running_mean;
            torch::Tensor running_var = instance_norm->running_var;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}