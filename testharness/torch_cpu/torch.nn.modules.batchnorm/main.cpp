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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for BatchNorm from the remaining data
        bool affine = (offset < Size) ? (Data[offset++] & 0x1) : true;
        bool track_running_stats = (offset < Size) ? (Data[offset++] & 0x1) : true;
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Clamp momentum to valid range [0, 1]
            momentum = std::max(0.0, std::min(1.0, momentum));
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Get the number of features from the input tensor
        int64_t num_features = 1;
        if (input.dim() >= 2) {
            num_features = input.size(1);
        } else if (input.dim() == 1) {
            num_features = input.size(0);
        }
        
        // Create BatchNorm modules for different dimensions
        if (input.dim() >= 2) {
            // Try BatchNorm1d
            if (input.dim() == 2 || input.dim() == 3) {
                torch::nn::BatchNorm1d bn1d(torch::nn::BatchNorm1dOptions(num_features)
                                            .eps(eps)
                                            .momentum(momentum)
                                            .affine(affine)
                                            .track_running_stats(track_running_stats));
                
                // Apply BatchNorm1d
                torch::Tensor output1d = bn1d->forward(input);
            }
            
            // Try BatchNorm2d
            if (input.dim() == 4) {
                torch::nn::BatchNorm2d bn2d(torch::nn::BatchNorm2dOptions(num_features)
                                            .eps(eps)
                                            .momentum(momentum)
                                            .affine(affine)
                                            .track_running_stats(track_running_stats));
                
                // Apply BatchNorm2d
                torch::Tensor output2d = bn2d->forward(input);
            }
            
            // Try BatchNorm3d
            if (input.dim() == 5) {
                torch::nn::BatchNorm3d bn3d(torch::nn::BatchNorm3dOptions(num_features)
                                            .eps(eps)
                                            .momentum(momentum)
                                            .affine(affine)
                                            .track_running_stats(track_running_stats));
                
                // Apply BatchNorm3d
                torch::Tensor output3d = bn3d->forward(input);
            }
        }
        
        // Try functional batch norm regardless of dimensions
        torch::Tensor weight;
        torch::Tensor bias;
        torch::Tensor running_mean;
        torch::Tensor running_var;
        
        if (affine) {
            weight = torch::ones({num_features});
            bias = torch::zeros({num_features});
        }
        
        if (track_running_stats) {
            running_mean = torch::zeros({num_features});
            running_var = torch::ones({num_features});
        }
        
        // Apply functional batch norm with options
        auto options = torch::nn::functional::BatchNormFuncOptions()
                          .weight(weight)
                          .bias(bias)
                          .training(!track_running_stats)
                          .momentum(momentum)
                          .eps(eps);
        
        torch::Tensor output_func = torch::nn::functional::batch_norm(
            input,
            running_mean,
            running_var,
            options
        );
        
        // Test with different training modes
        if (track_running_stats) {
            // Create a BatchNorm module
            torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(num_features)
                                      .eps(eps)
                                      .momentum(momentum)
                                      .affine(affine)
                                      .track_running_stats(true));
            
            // Test in training mode
            bn->train();
            torch::Tensor output_train = bn->forward(input);
            
            // Test in eval mode
            bn->eval();
            torch::Tensor output_eval = bn->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}