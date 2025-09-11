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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple neural network module
        torch::nn::Linear model(10, 5);
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape input if needed to match model's expected input shape
        if (input.dim() == 0) {
            input = input.reshape({1, 10});
        } else if (input.dim() == 1) {
            if (input.size(0) != 10) {
                input = input.reshape({-1}).slice(0, 0, std::min(static_cast<int64_t>(10), input.size(0)));
                input = torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions({0, 10 - input.size(0)}));
            }
            input = input.reshape({1, 10});
        } else {
            // For higher dimensions, reshape to have the last dimension as 10
            std::vector<int64_t> new_shape = {1, 10};
            input = input.reshape(new_shape);
        }
        
        // Convert input to float if needed
        if (input.scalar_type() != torch::kFloat && 
            input.scalar_type() != torch::kDouble) {
            input = input.to(torch::kFloat);
        }
        
        // Get model parameters
        auto params = model->parameters();
        
        // Create a vector of parameter values from the remaining data
        std::vector<torch::Tensor> param_values;
        for (auto& p : params) {
            if (offset < Size) {
                torch::Tensor param_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Reshape and convert to match parameter shape and type
                if (param_tensor.numel() > 0) {
                    param_tensor = param_tensor.to(p.dtype());
                    
                    // Try to reshape to match parameter shape
                    try {
                        param_tensor = param_tensor.reshape(p.sizes());
                    } catch (const std::exception&) {
                        // If reshape fails, create a new tensor with the right shape
                        param_tensor = torch::ones_like(p);
                    }
                    
                    param_values.push_back(param_tensor);
                } else {
                    // If empty tensor, use a default one
                    param_values.push_back(torch::ones_like(p));
                }
            } else {
                // If we run out of data, use default parameters
                param_values.push_back(torch::ones_like(p));
            }
        }
        
        // Test basic functionality since stateless is not available
        try {
            // Apply the model normally
            auto output = model->forward(input);
            
            // Test with modified parameters by copying them back
            if (!param_values.empty() && offset < Size) {
                // Modify one parameter value
                param_values[0] = param_values[0] * 2.0;
                
                // Copy parameters back to model
                auto model_params = model->parameters();
                auto param_iter = model_params.begin();
                for (size_t i = 0; i < param_values.size() && param_iter != model_params.end(); ++i, ++param_iter) {
                    param_iter->data().copy_(param_values[i]);
                }
                
                auto output2 = model->forward(input);
            }
            
            // Test with different input
            if (offset < Size) {
                torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
                input2 = input2.reshape({1, 10}).to(torch::kFloat);
                auto output3 = model->forward(input2);
            }
        } catch (const std::exception&) {
            // Catch exceptions from operations but continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
