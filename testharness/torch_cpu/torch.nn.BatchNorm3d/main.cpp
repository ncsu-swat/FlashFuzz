#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 5) {
            return 0;
        }
        
        // Create input tensor - must be 5D for BatchNorm3d (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 1 more byte for parameters
        if (offset >= Size) {
            return 0;
        }
        
        // Extract parameters for BatchNorm3d
        uint8_t param_byte = Data[offset++];
        
        // Parse other parameters from the fuzzer data
        bool affine = (param_byte & 0x01) != 0;
        bool track_running_stats = (param_byte & 0x02) != 0;
        double momentum = (param_byte & 0x04) ? 0.1 : 0.01;
        double eps = (param_byte & 0x08) ? 1e-5 : 1e-4;
        
        // Determine num_features - use a reasonable value derived from input
        int64_t num_features = 1;
        if (input.dim() >= 2) {
            num_features = input.size(1);
        } else if (input.dim() == 1 && input.numel() > 0) {
            // For 1D tensor, use a small portion as channels
            num_features = std::min(input.numel(), (int64_t)16);
        }
        
        // Clamp num_features to reasonable range
        if (num_features <= 0) {
            num_features = 1;
        }
        if (num_features > 256) {
            num_features = 256;  // Limit to avoid memory issues
        }
        
        // Create BatchNorm3d module
        torch::nn::BatchNorm3d bn(torch::nn::BatchNorm3dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // If input doesn't have 5 dimensions, reshape it to make it compatible
        if (input.dim() != 5) {
            // Inner try-catch for expected reshape failures
            try {
                int64_t total_elements = input.numel();
                if (total_elements == 0) {
                    return 0;  // Cannot reshape empty tensor
                }
                
                // Calculate dimensions: N=1, C=num_features, D, H, W
                // We need total_elements = 1 * num_features * D * H * W
                int64_t remaining = total_elements / num_features;
                if (remaining == 0 || total_elements % num_features != 0) {
                    // Adjust num_features to divide evenly
                    num_features = 1;
                    remaining = total_elements;
                    
                    // Recreate module with new num_features
                    bn = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(num_features)
                                                .eps(eps)
                                                .momentum(momentum)
                                                .affine(affine)
                                                .track_running_stats(track_running_stats));
                }
                
                // Factor remaining into D, H, W
                int64_t d = 1, h = 1, w = remaining;
                
                // Try to make dimensions more balanced
                for (int64_t i = 2; i * i <= remaining; i++) {
                    if (remaining % i == 0) {
                        d = i;
                        int64_t hw = remaining / i;
                        for (int64_t j = 2; j * j <= hw; j++) {
                            if (hw % j == 0) {
                                h = j;
                                w = hw / j;
                                break;
                            }
                        }
                        break;
                    }
                }
                
                input = input.reshape({1, num_features, d, h, w});
            } catch (...) {
                // Reshape failed, skip this input
                return 0;
            }
        } else {
            // Input is already 5D, ensure num_features matches
            int64_t actual_channels = input.size(1);
            if (actual_channels != num_features) {
                num_features = actual_channels;
                if (num_features <= 0) {
                    return 0;
                }
                bn = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(num_features)
                                            .eps(eps)
                                            .momentum(momentum)
                                            .affine(affine)
                                            .track_running_stats(track_running_stats));
            }
        }
        
        // Ensure input is float type for BatchNorm
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply BatchNorm3d in eval mode first (doesn't update running stats)
        bn->eval();
        torch::Tensor output_eval = bn->forward(input);
        
        // Test training mode
        bn->train();
        torch::Tensor output_train = bn->forward(input);
        
        // Access running mean and variance
        if (track_running_stats) {
            auto running_mean = bn->running_mean;
            auto running_var = bn->running_var;
            // Use the values to prevent optimization
            (void)running_mean.numel();
            (void)running_var.numel();
        }
        
        // Access weight and bias if affine is true
        if (affine) {
            auto weight = bn->weight;
            auto bias = bn->bias;
            // Use the values to prevent optimization
            (void)weight.numel();
            (void)bias.numel();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}