#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - should be 4D for BatchNorm2d (N, C, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least some data left for parameters
        if (offset >= Size) {
            return 0;
        }
        
        // Extract parameters for BatchNorm2d
        int64_t num_features = 0;
        
        // If input is 4D, use the number of channels (dim 1) as num_features
        if (input.dim() == 4) {
            num_features = input.size(1);
        } else {
            // For non-4D tensors, extract num_features from the remaining data
            if (offset + sizeof(int64_t) <= Size) {
                memcpy(&num_features, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Ensure num_features is positive and reasonable
                num_features = std::abs(num_features) % 1024 + 1;
            } else {
                // Default if not enough data
                num_features = 3;
            }
        }
        
        // Extract other BatchNorm2d parameters
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        // Extract eps if we have data
        if (offset + sizeof(double) <= Size) {
            memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive and not too small
            if (std::isnan(eps) || std::isinf(eps) || eps <= 0) {
                eps = 1e-5;
            }
        }
        
        // Extract momentum if we have data
        if (offset + sizeof(double) <= Size) {
            memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure momentum is in [0, 1]
            if (std::isnan(momentum) || std::isinf(momentum) || momentum < 0 || momentum > 1) {
                momentum = 0.1;
            }
        }
        
        // Extract boolean parameters if we have data
        if (offset < Size) {
            affine = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            track_running_stats = Data[offset++] & 0x1;
        }
        
        // Create BatchNorm2d module
        torch::nn::BatchNorm2d bn(torch::nn::BatchNorm2dOptions(num_features)
                                  .eps(eps)
                                  .momentum(momentum)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats));
        
        // Apply BatchNorm2d to the input tensor
        torch::Tensor output;
        
        // Try to run the operation
        output = bn->forward(input);
        
        // Access output to ensure computation is not optimized away
        auto sum = output.sum().item<float>();
        
        // Try to run backward pass if we have gradients
        if (output.requires_grad()) {
            auto grad_output = torch::ones_like(output);
            output.backward(grad_output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}