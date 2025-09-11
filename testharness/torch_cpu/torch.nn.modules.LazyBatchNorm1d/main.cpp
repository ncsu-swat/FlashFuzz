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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for BatchNorm1d from the remaining data
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
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
        
        if (offset < Size) {
            affine = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            track_running_stats = Data[offset++] & 0x1;
        }
        
        // Reshape input if needed to match BatchNorm1d requirements
        // BatchNorm1d expects input of shape [N, C, L]
        if (input.dim() == 1) {
            // Add batch and length dimensions
            input = input.unsqueeze(0).unsqueeze(2);
        } else if (input.dim() == 2) {
            // Add length dimension
            input = input.unsqueeze(2);
        } else if (input.dim() > 3) {
            // Flatten extra dimensions into the length dimension
            auto sizes = input.sizes().vec();
            int64_t new_length = 1;
            for (size_t i = 2; i < sizes.size(); ++i) {
                new_length *= sizes[i];
            }
            input = input.reshape({sizes[0], sizes[1], new_length});
        }
        
        // Get the number of features from the input tensor
        int64_t num_features = input.size(1);
        
        // Create BatchNorm1d module (using regular BatchNorm1d since LazyBatchNorm1d is not available)
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // Apply the BatchNorm1d operation
        torch::Tensor output = bn(input);
        
        // Access parameters
        if (bn->weight.defined()) {
            auto w = bn->weight.data();
        }
        if (bn->bias.defined()) {
            auto b = bn->bias.data();
        }
        if (bn->running_mean.defined()) {
            auto rm = bn->running_mean.data();
        }
        if (bn->running_var.defined()) {
            auto rv = bn->running_var.data();
        }
        
        // Test the module in training and eval modes
        bn->train();
        torch::Tensor train_output = bn(input);
        
        bn->eval();
        torch::Tensor eval_output = bn(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
