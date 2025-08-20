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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor for InstanceNorm3d
        // This should be a 5D tensor (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for InstanceNorm3d from the remaining data
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
        
        // Ensure input is 5D for InstanceNorm3d
        if (input.dim() < 5) {
            std::vector<int64_t> new_shape;
            for (int i = 0; i < 5; ++i) {
                if (i < input.dim()) {
                    new_shape.push_back(input.size(i));
                } else {
                    new_shape.push_back(1);
                }
            }
            input = input.reshape(new_shape);
        }
        
        int64_t num_features = input.size(1);
        
        // Create InstanceNorm3d module
        torch::nn::InstanceNorm3d norm(
            torch::nn::InstanceNorm3dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the operation
        torch::Tensor output = norm(input);
        
        // Force evaluation of lazy tensor
        output = output.clone();
        
        // Access some properties to ensure computation
        auto sizes = output.sizes();
        auto dtype = output.dtype();
        
        // Try to access the first element if tensor is not empty
        if (output.numel() > 0) {
            auto first_elem = output.flatten()[0].item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}