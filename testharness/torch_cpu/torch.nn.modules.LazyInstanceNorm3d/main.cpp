#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has 5 dimensions (N, C, D, H, W) for InstanceNorm3d
        // If not, reshape it to a valid 5D shape
        if (input.dim() != 5) {
            std::vector<int64_t> new_shape;
            
            if (input.dim() > 5) {
                // Flatten extra dimensions
                int64_t product = 1;
                for (int i = 5; i < input.dim(); i++) {
                    product *= input.size(i);
                }
                
                new_shape = {
                    input.size(0),
                    input.size(1),
                    input.size(2),
                    input.size(3),
                    input.size(4) * product
                };
            } else {
                // Add missing dimensions
                new_shape = {1, 1, 1, 1, 1};
                for (int i = 0; i < input.dim(); i++) {
                    new_shape[5 - input.dim() + i] = input.size(i);
                }
            }
            
            // Reshape the tensor
            input = input.reshape(new_shape);
        }
        
        // Ensure we have at least one channel
        if (input.size(1) == 0) {
            input = input.reshape({input.size(0), 1, input.size(2), input.size(3), input.size(4)});
        }
        
        // Extract parameters for InstanceNorm3d from the input data
        bool affine = offset < Size ? (Data[offset++] % 2 == 0) : true;
        bool track_running_stats = offset < Size ? (Data[offset++] % 2 == 0) : true;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive and not too small
            eps = std::abs(eps_raw);
            if (eps < 1e-10) eps = 1e-5;
        }
        
        if (offset + sizeof(float) <= Size) {
            float momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum_raw);
            momentum = momentum > 1.0 ? momentum - std::floor(momentum) : momentum;
        }
        
        // Create InstanceNorm3d module
        torch::nn::InstanceNorm3d norm(
            torch::nn::InstanceNorm3dOptions(input.size(1))
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the normalization
        torch::Tensor output = norm(input);
        
        // Force computation of lazy tensors
        output = output.clone();
        
        // Test edge cases with different parameters
        if (offset < Size) {
            // Try with different eps values
            double new_eps = 1e-10;
            torch::nn::InstanceNorm3d norm2(
                torch::nn::InstanceNorm3dOptions(input.size(1))
                    .eps(new_eps)
                    .momentum(momentum)
                    .affine(affine)
                    .track_running_stats(track_running_stats)
            );
            output = norm2(input);
            output = output.clone();
            
            // Try with different momentum values
            double new_momentum = 0.01;
            torch::nn::InstanceNorm3d norm3(
                torch::nn::InstanceNorm3dOptions(input.size(1))
                    .eps(eps)
                    .momentum(new_momentum)
                    .affine(!affine)  // Toggle affine
                    .track_running_stats(!track_running_stats)  // Toggle track_running_stats
            );
            output = norm3(input);
            output = output.clone();
        }
        
        // Test with eval mode
        norm.eval();
        output = norm(input);
        output = output.clone();
        
        // Test with train mode again
        norm.train();
        output = norm(input);
        output = output.clone();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
