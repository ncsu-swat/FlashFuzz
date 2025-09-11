#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight and bias tensors
        // For batch norm, weight and bias should have the same size as the number of channels
        // which is typically the second dimension (dim=1) for NCHW format
        int64_t num_features = 1;
        if (input.dim() > 1) {
            num_features = input.size(1);
        }
        
        // Create weight and bias with appropriate size
        torch::Tensor weight = torch::ones({num_features});
        torch::Tensor bias = torch::zeros({num_features});
        
        // Create running mean and running var tensors
        torch::Tensor running_mean = torch::zeros({num_features});
        torch::Tensor running_var = torch::ones({num_features});
        
        // Get training mode from input data if available
        bool training = true;
        if (offset < Size) {
            training = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Get momentum from input data if available
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is in valid range [0, 1]
            momentum = std::abs(momentum);
            momentum = momentum > 1.0 ? momentum - std::floor(momentum) : momentum;
        }
        
        // Get epsilon from input data if available
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure epsilon is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Apply native_batch_norm
        auto result = torch::native_batch_norm(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps
        );
        
        // Unpack the result (output, save_mean, save_var)
        torch::Tensor output = std::get<0>(result);
        torch::Tensor save_mean = std::get<1>(result);
        torch::Tensor save_var = std::get<2>(result);
        
        // Verify the output is not NaN or Inf
        if (output.defined() && !output.isnan().any().item<bool>() && !output.isinf().any().item<bool>()) {
            // Output is valid
        }
        
        // Try with different parameters if available
        if (offset < Size) {
            bool training2 = static_cast<bool>(Data[offset++] & 0x01);
            
            auto result2 = torch::native_batch_norm(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training2,
                momentum,
                eps
            );
            
            torch::Tensor output2 = std::get<0>(result2);
            
            if (output2.defined() && !output2.isnan().any().item<bool>() && !output2.isinf().any().item<bool>()) {
                // Output is valid
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
