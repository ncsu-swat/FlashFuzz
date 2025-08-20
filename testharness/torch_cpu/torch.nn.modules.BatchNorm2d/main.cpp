#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - should be 4D for BatchNorm2d (N, C, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we have more data, use it to configure BatchNorm2d parameters
        uint8_t num_features = 0;
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset + 1 < Size) {
            // Extract num_features from the data
            num_features = Data[offset++];
            // Ensure num_features is at least 1
            num_features = std::max(uint8_t(1), num_features);
        }
        
        if (offset + sizeof(double) <= Size) {
            // Extract eps from the data
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            // Avoid extremely small values that might cause numerical issues
            eps = std::max(1e-10, eps);
        }
        
        if (offset + sizeof(double) <= Size) {
            // Extract momentum from the data
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            momentum = momentum > 1.0 ? momentum - std::floor(momentum) : momentum;
        }
        
        if (offset < Size) {
            // Extract affine flag from the data
            affine = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            // Extract track_running_stats flag from the data
            track_running_stats = Data[offset++] & 0x1;
        }
        
        // If input is not 4D, reshape it to make it compatible with BatchNorm2d
        if (input.dim() != 4) {
            // Create a 4D tensor with at least one channel
            std::vector<int64_t> new_shape;
            
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, num_features, 1, 1]
                new_shape = {1, static_cast<int64_t>(num_features), 1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, num_features, 1, input.size(0)]
                new_shape = {1, static_cast<int64_t>(num_features), 1, std::max(int64_t(1), input.size(0))};
            } else if (input.dim() == 2) {
                // 2D tensor, reshape to [1, num_features, input.size(0), input.size(1)]
                new_shape = {1, static_cast<int64_t>(num_features), std::max(int64_t(1), input.size(0)), 
                             std::max(int64_t(1), input.size(1))};
            } else if (input.dim() == 3) {
                // 3D tensor, reshape to [1, num_features, input.size(1), input.size(2)]
                new_shape = {1, static_cast<int64_t>(num_features), std::max(int64_t(1), input.size(1)), 
                             std::max(int64_t(1), input.size(2))};
            } else {
                // More than 4D, take first 4 dimensions
                new_shape = {std::max(int64_t(1), input.size(0)), 
                             std::max(int64_t(1), static_cast<int64_t>(num_features)), 
                             std::max(int64_t(1), input.size(2)), 
                             std::max(int64_t(1), input.size(3))};
            }
            
            // Create a new tensor with the desired shape
            input = torch::ones(new_shape, input.options());
        }
        
        // If the second dimension (channels) doesn't match num_features, adjust it
        if (input.size(1) != num_features) {
            num_features = input.size(1);
        }
        
        // Create BatchNorm2d module
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // Apply BatchNorm2d to the input tensor
        torch::Tensor output = bn->forward(input);
        
        // Test the module in training and evaluation modes
        bn->train();
        torch::Tensor output_train = bn->forward(input);
        
        bn->eval();
        torch::Tensor output_eval = bn->forward(input);
        
        // Test with different data types if possible
        if (input.dtype() != torch::kFloat32 && input.dtype() != torch::kFloat) {
            // Try with float32
            torch::Tensor input_float = input.to(torch::kFloat32);
            torch::nn::BatchNorm2d bn_float(torch::nn::BatchNorm2dOptions(num_features)
                                           .eps(eps)
                                           .momentum(momentum)
                                           .affine(affine)
                                           .track_running_stats(track_running_stats));
            torch::Tensor output_float = bn_float->forward(input_float);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}