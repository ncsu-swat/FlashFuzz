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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for InstanceNorm1d
        // We need at least 5 bytes for the parameters
        if (offset + 5 > Size) {
            return 0;
        }
        
        // Get number of features
        int64_t num_features = 0;
        if (input.dim() > 0) {
            // For NCL format, features is the second dimension (C)
            if (input.dim() >= 2) {
                num_features = input.size(1);
            } else {
                // If only 1D, use the first dimension
                num_features = input.size(0);
            }
        } else {
            // For scalar tensor, default to a small number
            num_features = 1 + (Data[offset++] % 10);
        }
        
        // Extract other parameters
        bool affine = Data[offset++] % 2 == 1;
        bool track_running_stats = Data[offset++] % 2 == 1;
        
        // Extract epsilon (small value to prevent division by zero)
        double eps = 1e-5;
        if (offset + 2 <= Size) {
            uint16_t eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(eps_raw));
            offset += sizeof(eps_raw);
            // Convert to a small positive value
            eps = 1e-10 + (eps_raw % 1000) * 1e-6;
        }
        
        // Extract momentum
        double momentum = 0.1;
        if (offset + 2 <= Size) {
            uint16_t momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(momentum_raw));
            offset += sizeof(momentum_raw);
            // Convert to a value between 0 and 1
            momentum = static_cast<double>(momentum_raw % 1000) / 1000.0;
        }
        
        // Create InstanceNorm1d module
        torch::nn::InstanceNorm1d instance_norm(
            torch::nn::InstanceNorm1dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Ensure input has correct shape for InstanceNorm1d
        // InstanceNorm1d expects input of shape (N, C, L) or (C, L)
        if (input.dim() == 1) {
            // If 1D, reshape to (1, C, 1)
            input = input.reshape({1, input.size(0), 1});
        } else if (input.dim() == 2) {
            // If 2D, assume it's (N, C) and reshape to (N, C, 1)
            input = input.reshape({input.size(0), input.size(1), 1});
        } else if (input.dim() > 3) {
            // If more than 3D, flatten extra dimensions
            std::vector<int64_t> new_shape = {input.size(0), input.size(1), -1};
            input = input.reshape(new_shape);
        }
        
        // Apply the operation
        torch::Tensor output = instance_norm(input);
        
        // Force computation to ensure any errors are caught
        output = output.contiguous();
        
        // Access some elements to ensure computation is performed
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}