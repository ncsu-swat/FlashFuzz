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
        uint8_t num_features = 0;
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset < Size) {
            // Extract num_features from the data
            if (input.dim() >= 2) {
                // For BatchNorm1d, num_features should match the second dimension (C) of input
                num_features = input.size(1);
            } else if (input.dim() == 1) {
                // For 1D input, use the first dimension
                num_features = input.size(0);
            } else {
                // For scalar input, use a value from the data
                num_features = Data[offset++] % 64 + 1; // Ensure at least 1 feature
            }
            
            // Extract eps if we have more data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&eps, Data + offset, sizeof(double));
                offset += sizeof(double);
                // Ensure eps is positive and not too small
                eps = std::abs(eps);
                if (eps < 1e-10) eps = 1e-5;
            }
            
            // Extract momentum if we have more data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&momentum, Data + offset, sizeof(double));
                offset += sizeof(double);
                // Clamp momentum to [0, 1]
                momentum = std::max(0.0, std::min(1.0, momentum));
            }
            
            // Extract boolean parameters if we have more data
            if (offset < Size) {
                affine = (Data[offset++] % 2) == 1;
            }
            
            if (offset < Size) {
                track_running_stats = (Data[offset++] % 2) == 1;
            }
        }
        
        // Create BatchNorm1d module
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // Apply BatchNorm1d to the input tensor
        torch::Tensor output = bn->forward(input);
        
        // Ensure the output is valid by performing a simple operation
        auto sum = output.sum();
        
        // Test training mode
        bn->train();
        torch::Tensor train_output = bn->forward(input);
        
        // Test eval mode
        bn->eval();
        torch::Tensor eval_output = bn->forward(input);
        
        // Test with different batch sizes
        if (input.dim() >= 2 && input.size(0) > 1) {
            // Try with a smaller batch
            torch::Tensor smaller_batch = input.slice(0, 0, input.size(0)/2);
            torch::Tensor smaller_output = bn->forward(smaller_batch);
        }
        
        // Test with different data types if input is floating point
        if (input.scalar_type() == torch::kFloat || 
            input.scalar_type() == torch::kDouble) {
            // Try with a different dtype
            torch::Tensor input_other_dtype = input.to(input.scalar_type() == torch::kFloat ? 
                                                      torch::kDouble : torch::kFloat);
            torch::Tensor output_other_dtype = bn->forward(input_other_dtype);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}