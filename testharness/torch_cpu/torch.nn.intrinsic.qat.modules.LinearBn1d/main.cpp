#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        
        // Ensure input has at least 2 dimensions for Linear
        // First dimension is batch size, second is input features
        if (input.dim() < 2) {
            input = input.reshape({1, -1});
        }
        
        // Extract parameters for Linear from the remaining data
        int64_t in_features = input.size(-1);
        int64_t out_features = 1;
        
        // Parse out_features if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is positive and reasonable
            out_features = std::abs(out_features) % 100 + 1;
        }
        
        // Parse bias flag if we have more data
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create the Linear module (using standard Linear instead of intrinsic)
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Set module to training mode
        linear->train();
        
        // Apply the module to the input tensor
        torch::Tensor output = linear->forward(input);
        
        // Try to access and modify some parameters to test more functionality
        if (offset < Size) {
            // Modify weight if we have more data
            auto weight = linear->weight;
            if (weight.defined()) {
                // Scale the weight by a small factor
                float scale = static_cast<float>(Data[offset++]) / 255.0f;
                linear->weight = weight * scale;
            }
        }
        
        if (offset < Size) {
            // Modify bias if it exists and we have more data
            auto bias_param = linear->bias;
            if (bias && bias_param.defined()) {
                // Add a small value to bias
                float bias_add = static_cast<float>(Data[offset++]) / 255.0f;
                linear->bias = bias_param + bias_add;
            }
        }
        
        // Try running backward pass if we have a non-empty output
        if (output.numel() > 0) {
            auto output_grad = torch::ones_like(output);
            output.backward(output_grad);
        }
        
        // Run forward again to test modified parameters
        output = linear->forward(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
