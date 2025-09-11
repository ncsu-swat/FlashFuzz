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
        
        // Extract parameters for Linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Determine in_features from input tensor
        if (input.dim() >= 2) {
            in_features = input.size(-1);
        } else if (input.dim() == 1) {
            in_features = input.size(0);
        } else {
            // For scalar tensors, use a default value
            in_features = 4;
        }
        
        // Get out_features from remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make out_features reasonable but allow edge cases
            out_features = std::abs(out_features) % 128 + 1;
        } else {
            out_features = 4; // Default value
        }
        
        // Get bias flag from remaining data
        if (offset < Size) {
            bias = Data[offset++] & 0x1; // Use lowest bit to determine bias
        }
        
        // Create regular Linear module (QAT modules are not available in C++ frontend)
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Set module to training mode
        linear->train();
        
        // Try different input shapes
        torch::Tensor output;
        if (input.dim() == 0) {
            // For scalar input, reshape to make it compatible
            input = input.reshape({1, in_features});
            output = linear->forward(input);
        } else if (input.dim() == 1) {
            // For 1D input, can be treated as a single feature vector
            if (input.size(0) == in_features) {
                output = linear->forward(input);
            } else {
                // Reshape to make compatible
                input = input.reshape({1, in_features});
                output = linear->forward(input);
            }
        } else {
            // For 2D+ inputs, ensure last dimension matches in_features
            std::vector<int64_t> new_shape = input.sizes().vec();
            if (new_shape.back() != in_features) {
                new_shape.back() = in_features;
                input = input.reshape(new_shape);
            }
            output = linear->forward(input);
        }
        
        // Test with different dtypes if possible
        if (input.scalar_type() != torch::kFloat) {
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::Tensor float_output = linear->forward(float_input);
        }
        
        // Test backward pass if using floating point
        if (input.scalar_type() == torch::kFloat || 
            input.scalar_type() == torch::kDouble) {
            
            // Make input require gradients
            input = input.detach().requires_grad_(true);
            
            // Forward pass
            output = linear->forward(input);
            
            // Create a scalar loss and backpropagate
            torch::Tensor loss = output.sum();
            loss.backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
