#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Linear layer
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in_features from input tensor if possible
        if (input.dim() >= 1) {
            in_features = input.size(-1);
        } else {
            // For scalar input, use a small value
            in_features = 4;
        }
        
        // Get out_features from remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make out_features reasonable but allow edge cases
            out_features = std::abs(out_features) % 128 + 1;
        } else {
            out_features = 4;
        }
        
        // Get bias parameter
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create a Linear module and BatchNorm1d module separately
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(out_features));
        
        // Set modules to training mode
        linear->train();
        bn->train();
        
        // Get momentum and eps parameters for batch norm
        double momentum = 0.1;
        double eps = 1e-5;
        
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
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Set batch norm parameters
        bn->options.momentum(momentum);
        bn->options.eps(eps);
        
        // Reshape input if needed to match expected dimensions for Linear + BatchNorm1d
        // Linear expects input of shape (N, in_features) or (in_features)
        if (input.dim() == 0) {
            // Scalar input - reshape to 1D
            input = input.reshape({1});
        }
        
        if (input.dim() == 1) {
            // If 1D tensor, ensure it has in_features elements
            if (input.size(0) != in_features) {
                input = input.reshape({1, -1});
                if (input.size(1) != in_features) {
                    // Pad or truncate to match in_features
                    torch::Tensor resized = torch::zeros({1, in_features}, input.options());
                    int64_t copy_size = std::min(input.size(1), in_features);
                    resized.slice(1, 0, copy_size).copy_(input.slice(1, 0, copy_size));
                    input = resized;
                }
            } else {
                // Correct size but need to add batch dimension
                input = input.unsqueeze(0);
            }
        } else if (input.dim() >= 2) {
            // For multi-dimensional tensors, ensure last dimension is in_features
            if (input.size(-1) != in_features) {
                // Reshape to have correct last dimension
                std::vector<int64_t> new_shape;
                int64_t total_elements = 1;
                for (int i = 0; i < input.dim() - 1; i++) {
                    total_elements *= input.size(i);
                    new_shape.push_back(input.size(i));
                }
                new_shape.push_back(in_features);
                
                // Create a new tensor with the right shape
                torch::Tensor resized = torch::zeros(new_shape, input.options());
                
                // Try to preserve as much data as possible
                int64_t copy_size = std::min(input.size(-1), in_features);
                
                // Create a view that matches the original tensor's shape except for the last dimension
                std::vector<int64_t> view_shape = new_shape;
                view_shape.back() = copy_size;
                
                // Copy data from the original tensor to the resized tensor
                resized.slice(-1, 0, copy_size).copy_(input.slice(-1, 0, copy_size).reshape(view_shape));
                input = resized;
            }
        }
        
        // Apply the linear layer followed by batch norm
        torch::Tensor linear_output = linear->forward(input);
        torch::Tensor output = bn->forward(linear_output);
        
        // Test with different configurations
        if (offset + 2 < Size) {
            // Get scale and zero_point for testing
            double scale = std::abs(static_cast<double>(Data[offset++]) / 255.0) + 1e-5;
            int64_t zero_point = static_cast<int64_t>(Data[offset++]) % 256;
            
            // Test with eval mode
            linear->eval();
            bn->eval();
            
            torch::Tensor eval_linear_output = linear->forward(input);
            torch::Tensor eval_output = bn->forward(eval_linear_output);
        }
        
        // Test with frozen parameters
        linear->weight.requires_grad_(false);
        if (linear->bias.defined()) {
            linear->bias.requires_grad_(false);
        }
        bn->weight.requires_grad_(false);
        bn->bias.requires_grad_(false);
        
        torch::Tensor frozen_linear_output = linear->forward(input);
        torch::Tensor frozen_output = bn->forward(frozen_linear_output);
        
        // Test with different batch sizes
        if (input.dim() >= 2 && input.size(0) > 1) {
            torch::Tensor single_sample = input.slice(0, 0, 1);
            torch::Tensor single_linear_output = linear->forward(single_sample);
            torch::Tensor single_output = bn->forward(single_linear_output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}