#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create grad_output tensor
        torch::Tensor grad_output = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a tensor with compatible shape
            if (grad_output.dim() >= 2) {
                auto grad_shape = grad_output.sizes();
                std::vector<int64_t> input_shape;
                input_shape.push_back(grad_shape[0]);  // batch size
                input_shape.push_back(grad_shape[1]);  // input features
                input = torch::ones(input_shape, grad_output.options());
            } else {
                // For low-dimensional inputs, create a simple tensor
                input = torch::ones({1, 1}, grad_output.options());
            }
        }
        
        // Create weight tensor
        torch::Tensor weight;
        if (input.dim() >= 2 && grad_output.dim() >= 2) {
            // For typical case: weight is (output_features, input_features)
            std::vector<int64_t> weight_shape = {grad_output.size(1), input.size(1)};
            weight = torch::ones(weight_shape, input.options());
        } else {
            // Fallback for edge cases
            weight = torch::ones({1, 1}, input.options());
        }
        
        // Try different values for bias_defined
        bool bias_defined = (offset < Size && Data[offset++] % 2 == 0);
        
        try {
            // Call mkldnn_linear_backward_weights
            auto result = torch::mkldnn_linear_backward_weights(
                grad_output,
                input,
                weight,
                bias_defined
            );
            
            // Access the results to ensure they're computed
            if (bias_defined) {
                auto& grad_weight = std::get<0>(result);
                auto& grad_bias = std::get<1>(result);
                
                // Force evaluation
                auto sum_weight = grad_weight.sum().item<float>();
                auto sum_bias = grad_bias.sum().item<float>();
                
                // Prevent compiler from optimizing away
                if (sum_weight == -999999.0f && sum_bias == -999999.0f) {
                    throw std::runtime_error("Unreachable");
                }
            } else {
                auto& grad_weight = std::get<0>(result);
                
                // Force evaluation
                auto sum_weight = grad_weight.sum().item<float>();
                
                // Prevent compiler from optimizing away
                if (sum_weight == -999999.0f) {
                    throw std::runtime_error("Unreachable");
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and okay
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}