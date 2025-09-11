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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a dynamic quantized linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        if (input_tensor.dim() >= 2) {
            in_features = input_tensor.size(-1);
            out_features = in_features > 0 ? (in_features % 8) + 1 : 1;
        } else if (input_tensor.dim() == 1) {
            in_features = input_tensor.size(0);
            out_features = in_features > 0 ? (in_features % 8) + 1 : 1;
        } else {
            in_features = 1;
            out_features = 1;
        }
        
        // Create a linear module
        torch::nn::Linear linear_module(in_features, out_features);
        
        // Try different quantization configurations
        bool has_bias = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Create model
        torch::nn::Sequential model;
        model->push_back(linear_module);
        
        // Try to reshape input tensor if needed to match module expectations
        if (input_tensor.dim() == 0) {
            input_tensor = input_tensor.reshape({1, in_features});
        } else if (input_tensor.dim() == 1) {
            input_tensor = input_tensor.reshape({1, input_tensor.size(0)});
        } else if (input_tensor.dim() > 2) {
            // Keep the batch dimension, flatten the rest into features
            int64_t batch_size = input_tensor.size(0);
            input_tensor = input_tensor.reshape({batch_size, -1});
            
            // If the reshaped tensor's second dimension doesn't match in_features,
            // we need to adjust it
            if (input_tensor.size(1) != in_features) {
                if (input_tensor.size(1) > 0) {
                    // Slice or pad as needed
                    if (input_tensor.size(1) > in_features) {
                        input_tensor = input_tensor.slice(1, 0, in_features);
                    } else {
                        auto padding = torch::zeros({batch_size, in_features - input_tensor.size(1)}, 
                                                   input_tensor.options());
                        input_tensor = torch::cat({input_tensor, padding}, 1);
                    }
                } else {
                    // Handle zero-sized dimension by creating a new tensor
                    input_tensor = torch::zeros({batch_size, in_features}, input_tensor.options());
                }
            }
        }
        
        // Forward pass through the model
        try {
            torch::Tensor output = model->forward(input_tensor);
        } catch (...) {
            // If forward fails, try with a different input shape
            try {
                // Create a tensor with the exact expected shape
                torch::Tensor fallback_input = torch::ones({1, in_features}, input_tensor.options());
                torch::Tensor output = model->forward(fallback_input);
            } catch (...) {
                // If that also fails, just continue
            }
        }
        
        // Try dynamic quantization using torch::jit
        try {
            torch::jit::script::Module scripted_model = torch::jit::trace(model, input_tensor);
            auto quantized_model = torch::jit::quantized::quantize_dynamic(scripted_model);
            torch::Tensor quantized_output = quantized_model.forward({input_tensor}).toTensor();
        } catch (...) {
            // If quantization fails, just continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
