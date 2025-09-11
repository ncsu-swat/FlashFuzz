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
        torch::Tensor weight, bias;
        if (offset < Size - 2) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
            if (offset < Size - 2) {
                bias = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Create a default bias tensor if we don't have enough data
                bias = torch::ones_like(weight);
            }
        } else {
            // Create default weight and bias tensors if we don't have enough data
            int64_t num_features = input.size(1);
            weight = torch::ones({num_features}, input.options());
            bias = torch::zeros({num_features}, input.options());
        }
        
        // Create running_mean and running_var tensors
        torch::Tensor running_mean, running_var;
        if (offset < Size - 2) {
            running_mean = fuzzer_utils::createTensor(Data, Size, offset);
            if (offset < Size - 2) {
                running_var = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                running_var = torch::ones_like(running_mean);
            }
        } else {
            // Create default running_mean and running_var tensors
            int64_t num_features = input.size(1);
            running_mean = torch::zeros({num_features}, input.options());
            running_var = torch::ones({num_features}, input.options());
        }
        
        // Get training mode and momentum from the input data
        bool training = true;
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset < Size) {
            training = Data[offset++] % 2 == 0;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is in valid range [0, 1]
            momentum = std::abs(momentum);
            momentum = momentum > 1.0 ? momentum - std::floor(momentum) : momentum;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0) eps = 1e-5;
        }
        
        // Ensure input has at least 2 dimensions for batch norm
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0);
            } else {
                input = input.unsqueeze(0);
            }
        }
        
        // Ensure weight, bias, running_mean, and running_var have the correct shape
        int64_t num_features = input.size(1);
        if (weight.dim() != 1 || weight.size(0) != num_features) {
            weight = torch::ones({num_features}, input.options());
        }
        if (bias.dim() != 1 || bias.size(0) != num_features) {
            bias = torch::zeros({num_features}, input.options());
        }
        if (running_mean.dim() != 1 || running_mean.size(0) != num_features) {
            running_mean = torch::zeros({num_features}, input.options());
        }
        if (running_var.dim() != 1 || running_var.size(0) != num_features) {
            running_var = torch::ones({num_features}, input.options());
        }
        
        // Apply miopen_batch_norm
        auto result = torch::miopen_batch_norm(
            input, weight, bias, running_mean, running_var,
            training, momentum, eps
        );
        
        // Access the output to ensure the operation is not optimized away
        torch::Tensor output = std::get<0>(result);
        torch::Tensor save_mean = std::get<1>(result);
        torch::Tensor save_var = std::get<2>(result);
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum().item<float>();
        if (std::isnan(sum) || std::isinf(sum)) {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
