#include "fuzzer_utils.h"
#include <iostream>

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
            // Ensure eps is positive and reasonable
            eps = std::abs(eps);
            if (eps < 1e-10 || std::isnan(eps) || std::isinf(eps)) eps = 1e-5;
        }
        
        // BatchNorm requires at least 2D input with batch dimension
        if (input.dim() < 2) {
            return 0;
        }
        
        // Get the number of features from the input tensor (channel dimension)
        int64_t num_features = input.size(1);
        
        // Ensure num_features is valid
        if (num_features <= 0 || num_features > 10000) {
            return 0;
        }
        
        // Try BatchNorm1d for 2D or 3D input
        if (input.dim() == 2 || input.dim() == 3) {
            try {
                torch::nn::BatchNorm1d bn1d(torch::nn::BatchNorm1dOptions(num_features)
                                            .eps(eps)
                                            .momentum(momentum)
                                            .affine(affine)
                                            .track_running_stats(track_running_stats));
                
                bn1d->train();
                torch::Tensor output1d_train = bn1d->forward(input);
                
                bn1d->eval();
                torch::Tensor output1d_eval = bn1d->forward(input);
            } catch (...) {
                // Expected failures for invalid shapes/sizes
            }
        }
        
        // Try BatchNorm2d for 4D input
        if (input.dim() == 4) {
            try {
                torch::nn::BatchNorm2d bn2d(torch::nn::BatchNorm2dOptions(num_features)
                                            .eps(eps)
                                            .momentum(momentum)
                                            .affine(affine)
                                            .track_running_stats(track_running_stats));
                
                bn2d->train();
                torch::Tensor output2d_train = bn2d->forward(input);
                
                bn2d->eval();
                torch::Tensor output2d_eval = bn2d->forward(input);
            } catch (...) {
                // Expected failures for invalid shapes/sizes
            }
        }
        
        // Try BatchNorm3d for 5D input
        if (input.dim() == 5) {
            try {
                torch::nn::BatchNorm3d bn3d(torch::nn::BatchNorm3dOptions(num_features)
                                            .eps(eps)
                                            .momentum(momentum)
                                            .affine(affine)
                                            .track_running_stats(track_running_stats));
                
                bn3d->train();
                torch::Tensor output3d_train = bn3d->forward(input);
                
                bn3d->eval();
                torch::Tensor output3d_eval = bn3d->forward(input);
            } catch (...) {
                // Expected failures for invalid shapes/sizes
            }
        }
        
        // Try functional batch norm
        try {
            torch::Tensor weight;
            torch::Tensor bias;
            torch::Tensor running_mean;
            torch::Tensor running_var;
            
            if (affine) {
                weight = torch::ones({num_features});
                bias = torch::zeros({num_features});
            }
            
            running_mean = torch::zeros({num_features});
            running_var = torch::ones({num_features});
            
            // Test in training mode
            auto options_train = torch::nn::functional::BatchNormFuncOptions()
                              .weight(weight)
                              .bias(bias)
                              .training(true)
                              .momentum(momentum)
                              .eps(eps);
            
            torch::Tensor output_func_train = torch::nn::functional::batch_norm(
                input,
                running_mean,
                running_var,
                options_train
            );
            
            // Test in eval mode
            auto options_eval = torch::nn::functional::BatchNormFuncOptions()
                              .weight(weight)
                              .bias(bias)
                              .training(false)
                              .momentum(momentum)
                              .eps(eps);
            
            torch::Tensor output_func_eval = torch::nn::functional::batch_norm(
                input,
                running_mean,
                running_var,
                options_eval
            );
        } catch (...) {
            // Expected failures for invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}