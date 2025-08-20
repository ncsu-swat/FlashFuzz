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
        
        // Extract parameters for BatchNorm1d from the remaining data
        int64_t num_features = 0;
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset < Size) {
            // Extract num_features from the data
            // For BatchNorm1d, num_features should match the second dimension (channels)
            // If input is 2D: [batch_size, num_features]
            // If input is 3D: [batch_size, num_features, length]
            if (input.dim() >= 2) {
                num_features = input.size(1);
            } else if (input.dim() == 1) {
                // For 1D input, use the first dimension
                num_features = input.size(0);
            } else {
                // For 0D input (scalar), use a small value
                num_features = 1 + (Data[offset] % 10);
            }
            offset++;
        }
        
        // Extract eps parameter
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive and not too small
            eps = std::abs(eps);
            if (eps < 1e-10) eps = 1e-5;
        }
        
        // Extract momentum parameter
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is in [0, 1]
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Extract boolean parameters
        if (offset < Size) {
            affine = (Data[offset] % 2) == 1;
            offset++;
        }
        
        if (offset < Size) {
            track_running_stats = (Data[offset] % 2) == 1;
            offset++;
        }
        
        // Create the BatchNorm1d module (PyTorch C++ doesn't have LazyBatchNorm1d)
        torch::nn::BatchNorm1d batch_norm(
            torch::nn::BatchNorm1dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the batch norm operation
        torch::Tensor output = batch_norm->forward(input);
        
        // Force materialization of the tensor
        output = output.clone();
        
        // Access some properties to ensure computation
        auto sizes = output.sizes();
        auto dtype = output.dtype();
        
        // Try to access the running stats if they're being tracked
        if (track_running_stats) {
            auto running_mean = batch_norm->running_mean;
            auto running_var = batch_norm->running_var;
        }
        
        // Try to access the learnable parameters if affine is true
        if (affine) {
            auto weight = batch_norm->weight;
            auto bias = batch_norm->bias;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}