#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 5 dimensions (N, C, D, H, W) for BNReLU3d
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            int64_t total_elements = input.numel();
            
            // Extract parameters for BNReLU3d from the remaining data
            uint8_t num_features = 0;
            if (offset < Size) {
                num_features = Data[offset++] % 64 + 1; // Ensure at least 1 feature
            } else {
                num_features = 3; // Default
            }
            
            // Calculate dimensions for reshaping
            int64_t batch_size = 1;
            int64_t depth = 1;
            int64_t height = 1;
            int64_t width = 1;
            
            // Calculate remaining dimensions to distribute elements
            int64_t remaining_elements = total_elements / num_features;
            
            if (remaining_elements > 0) {
                batch_size = std::max<int64_t>(1, remaining_elements % 8 + 1);
                remaining_elements /= batch_size;
                
                if (remaining_elements > 0) {
                    depth = std::max<int64_t>(1, remaining_elements % 4 + 1);
                    remaining_elements /= depth;
                    
                    if (remaining_elements > 0) {
                        height = std::max<int64_t>(1, remaining_elements % 4 + 1);
                        width = std::max<int64_t>(1, remaining_elements / height);
                    }
                }
            }
            
            // Reshape the tensor to 5D
            if (total_elements > 0) {
                input = input.reshape({batch_size, num_features, depth, height, width});
            } else {
                // Create a small tensor if input is empty
                input = torch::ones({1, num_features, 1, 1, 1}, input.options());
            }
        }
        
        // Extract parameters for BNReLU3d
        int64_t num_features = input.size(1); // Number of channels
        
        // Create BNReLU3d module
        auto bn = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(num_features));
        auto relu = torch::nn::ReLU();
        
        // Create BNReLU3d by sequential composition
        auto bnrelu3d = torch::nn::Sequential(bn, relu);
        
        // Extract additional parameters from remaining data
        if (offset + 3 < Size) {
            // Set training mode based on input data
            bool training_mode = (Data[offset++] % 2) == 1;
            bnrelu3d->train(training_mode);
            
            // Set momentum and eps if data available
            double momentum = 0.1;
            double eps = 1e-5;
            
            if (offset + sizeof(float) <= Size) {
                float momentum_raw;
                std::memcpy(&momentum_raw, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Clamp momentum to valid range [0, 1]
                momentum = std::abs(momentum_raw) / (1.0 + std::abs(momentum_raw));
            }
            
            if (offset + sizeof(float) <= Size) {
                float eps_raw;
                std::memcpy(&eps_raw, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Ensure eps is positive
                eps = std::abs(eps_raw) + 1e-10;
            }
            
            // Update BatchNorm3d parameters
            bn->options.momentum(momentum);
            bn->options.eps(eps);
        }
        
        // Apply BNReLU3d to input tensor
        torch::Tensor output = bnrelu3d->forward(input);
        
        // Verify output has same shape as input
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape doesn't match input shape");
        }
        
        // Verify all values in output are non-negative (ReLU effect)
        if (torch::any(output < 0).item<bool>()) {
            throw std::runtime_error("Output contains negative values after ReLU");
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}