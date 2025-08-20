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
        
        // Extract parameters for Linear layer
        // We need at least 2 bytes for in_features and out_features
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get in_features - use the first dimension of input if it exists, otherwise use data
        int64_t in_features = 1;
        if (input.dim() > 0) {
            in_features = input.size(-1);
        } else {
            in_features = static_cast<int64_t>(Data[offset++]) % 128 + 1;
        }
        
        // Get out_features from data
        int64_t out_features = static_cast<int64_t>(Data[offset++]) % 128 + 1;
        
        // Get bias flag if we have more data
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 1;
        }
        
        // Create Linear layer using LinearOptions
        torch::nn::LinearOptions options(in_features, out_features);
        options.bias(bias);
        torch::nn::Linear linear(options);
        
        // If we have more data, use it to initialize weights and bias
        if (offset + 2 < Size) {
            // Initialize weights with a small value to avoid numerical instability
            float weight_init = static_cast<float>(Data[offset++]) / 255.0f;
            linear->weight.data().fill_(weight_init);
            
            if (bias) {
                float bias_init = static_cast<float>(Data[offset++]) / 255.0f;
                linear->bias.data().fill_(bias_init);
            }
        }
        
        // Apply the linear layer to the input tensor
        torch::Tensor output;
        
        // Handle different input dimensions
        if (input.dim() == 0) {
            // For scalar input, reshape to 1D tensor with one element
            output = linear(input.reshape({1}));
        } else if (input.dim() == 1) {
            // For 1D input, check if size matches in_features
            if (input.size(0) != in_features) {
                // Reshape to match expected input size
                output = linear(input.reshape({1, in_features}));
            } else {
                output = linear(input);
            }
        } else {
            // For multi-dimensional input, ensure last dimension matches in_features
            auto input_sizes = input.sizes().vec();
            if (input_sizes.back() != in_features) {
                // Reshape the tensor to have the correct last dimension
                std::vector<int64_t> new_sizes = input_sizes;
                new_sizes.back() = in_features;
                output = linear(input.reshape(new_sizes));
            } else {
                output = linear(input);
            }
        }
        
        // Access output elements to ensure computation is performed
        float sum = output.sum().item<float>();
        
        // Try backward pass if we have enough data
        if (offset < Size && (Data[offset++] & 1)) {
            output.backward(torch::ones_like(output));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}