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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor - must be 5D for InstanceNorm3d (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least some data left for parameters
        if (offset >= Size - 5) {
            return 0;
        }
        
        // Extract parameters for InstanceNorm3d
        int64_t num_features = 0;
        
        // If input is 5D, use the channel dimension (dim 1) as num_features
        if (input.dim() == 5 && input.size(1) > 0) {
            num_features = input.size(1);
        } else {
            // For non-5D tensors, create a random number of features
            uint8_t num_features_byte = Data[offset++];
            num_features = (num_features_byte % 64) + 1; // 1-64 features
        }
        
        // Extract other parameters
        bool affine = Data[offset++] & 1;
        bool track_running_stats = Data[offset++] & 1;
        
        // Extract eps parameter (small positive value for numerical stability)
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive and not too large
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
            if (eps > 1.0) eps = 1.0;
        }
        
        // Extract momentum parameter
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure momentum is in [0, 1]
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = 1.0;
        }
        
        // Create InstanceNorm3d module
        torch::nn::InstanceNorm3d instance_norm(
            torch::nn::InstanceNorm3dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply InstanceNorm3d to the input tensor
        torch::Tensor output;
        
        // Set module to evaluation mode if we want to test that behavior
        if (offset < Size && (Data[offset++] & 1)) {
            instance_norm->eval();
        }
        
        // Apply the module
        output = instance_norm->forward(input);
        
        // Perform a simple operation on the output to ensure it's used
        auto sum = output.sum();
        
        // Optionally test backward pass if we have a floating point tensor
        if (input.scalar_type() == torch::kFloat || 
            input.scalar_type() == torch::kDouble || 
            input.scalar_type() == torch::kHalf) {
            
            if (offset < Size && (Data[offset++] & 1)) {
                // Make input require gradients
                auto input_with_grad = input.detach().requires_grad_(true);
                
                // Forward pass
                auto output_for_backward = instance_norm->forward(input_with_grad);
                
                // Backward pass
                output_for_backward.sum().backward();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}