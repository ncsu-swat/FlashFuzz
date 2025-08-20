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
        
        // Create input tensor - must be 5D for BatchNorm3d (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor has 5 dimensions for BatchNorm3d
        // If not, reshape it to a valid 5D shape
        if (input.dim() != 5) {
            std::vector<int64_t> new_shape;
            if (input.dim() > 5) {
                // Collapse extra dimensions
                int64_t product = 1;
                for (int i = 0; i < input.dim() - 4; i++) {
                    product *= input.size(i);
                }
                new_shape = {product, 2, 2, 2, 2};
            } else {
                // Add missing dimensions
                new_shape = {1, 1, 1, 1, 1};
                for (int i = 0; i < input.dim(); i++) {
                    new_shape[i] = input.size(i);
                }
            }
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for BatchNorm3d
        uint8_t param_byte = 0;
        if (offset < Size) {
            param_byte = Data[offset++];
        }
        
        // Parse num_features - use the channel dimension (dim 1) of the input tensor
        int64_t num_features = input.size(1);
        
        // Parse eps - small value added to denominator for numerical stability
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Parse momentum - value used for running_mean and running_var computation
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Clamp momentum to [0, 1]
            momentum = std::max(0.0, std::min(1.0, momentum));
        }
        
        // Parse affine flag - whether to use learnable affine parameters
        bool affine = (param_byte & 0x01) != 0;
        
        // Parse track_running_stats flag - whether to track running stats
        bool track_running_stats = (param_byte & 0x02) != 0;
        
        // Create BatchNorm3d module (LazyBatchNorm3d doesn't exist, use regular BatchNorm3d)
        torch::nn::BatchNorm3d bn(torch::nn::BatchNorm3dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // Apply the BatchNorm3d operation
        torch::Tensor output = bn->forward(input);
        
        // Force materialization of the tensor
        output = output.clone();
        
        // Access some properties to ensure computation
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        // Access module parameters if they exist
        if (affine) {
            auto weight = bn->weight;
            auto bias = bn->bias;
        }
        
        // Access running stats if tracking
        if (track_running_stats) {
            auto running_mean = bn->running_mean;
            auto running_var = bn->running_var;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}