#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 2 dimensions for InstanceNorm1d
        // If not, reshape it to have a batch dimension and a channel dimension
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                input = input.reshape({1, 1});
            } else if (input.dim() == 1) {
                input = input.reshape({1, input.size(0)});
            }
        }
        
        // Extract parameters for InstanceNorm1d from the remaining data
        bool affine = true;
        bool track_running_stats = true;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 2 <= Size) {
            affine = Data[offset++] & 1;
            track_running_stats = Data[offset++] & 1;
        }
        
        if (offset + sizeof(double) <= Size) {
            memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Get the number of features (channels) from the input tensor
        int64_t num_features = input.size(1);
        
        // Create InstanceNorm1d module
        auto instance_norm = torch::nn::InstanceNorm1d(
            torch::nn::InstanceNorm1dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the module to the input tensor
        torch::Tensor output = instance_norm->forward(input);
        
        // Ensure the output is valid
        if (output.isnan().any().item<bool>() || output.isinf().any().item<bool>()) {
            return 0;
        }
        
        // Test the module in eval mode
        instance_norm->eval();
        torch::Tensor eval_output = instance_norm->forward(input);
        
        // Test the module with different batch sizes
        if (input.size(0) > 1 && input.size(0) % 2 == 0) {
            torch::Tensor half_input = input.slice(0, 0, input.size(0) / 2);
            torch::Tensor half_output = instance_norm->forward(half_input);
        }
        
        // Test with different input shapes if possible
        if (input.dim() > 2) {
            // Reshape to keep batch and channel dimensions but flatten others
            std::vector<int64_t> new_shape = {input.size(0), input.size(1), -1};
            torch::Tensor reshaped_input = input.reshape(new_shape);
            torch::Tensor reshaped_output = instance_norm->forward(reshaped_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}