#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// Custom Sequential wrapper to avoid template issues
class CustomSequential : public torch::nn::Module {
public:
    CustomSequential() = default;
    
    void push_back(torch::nn::AnyModule module) {
        modules_.push_back(module);
        register_module("layer_" + std::to_string(modules_.size() - 1), module);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        for (auto& module : modules_) {
            x = module.forward(x);
        }
        return x;
    }
    
private:
    std::vector<torch::nn::AnyModule> modules_;
};

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a Sequential model with various layers
        torch::nn::Sequential model;
        
        // Use remaining bytes to determine model structure
        if (offset < Size) {
            uint8_t layer_selector = Data[offset++] % 5;
            
            // Add a Linear layer
            int64_t in_features = input.dim() > 0 ? input.size(-1) : 1;
            int64_t out_features = 10;
            
            if (offset + 1 < Size) {
                out_features = (Data[offset] % 32) + 1;
                offset++;
            }
            
            model->push_back(torch::nn::Linear(in_features, out_features));
            
            // Add activation function based on selector
            switch (layer_selector) {
                case 0:
                    model->push_back(torch::nn::ReLU());
                    break;
                case 1:
                    model->push_back(torch::nn::Sigmoid());
                    break;
                case 2:
                    model->push_back(torch::nn::Tanh());
                    break;
                case 3:
                    model->push_back(torch::nn::GELU());
                    break;
                case 4:
                    model->push_back(torch::nn::Dropout(0.5));
                    break;
            }
            
            // Add another Linear layer
            model->push_back(torch::nn::Linear(out_features, 1));
        } else {
            // Default model if not enough bytes
            int64_t in_features = input.dim() > 0 ? input.size(-1) : 1;
            model->push_back(torch::nn::Linear(in_features, 5));
            model->push_back(torch::nn::ReLU());
            model->push_back(torch::nn::Linear(5, 1));
        }
        
        // Try different modes
        if (offset < Size) {
            uint8_t mode_selector = Data[offset++];
            if (mode_selector % 2 == 0) {
                model->eval();
            } else {
                model->train();
            }
        }
        
        // Apply the model to the input tensor
        torch::Tensor output;
        
        // Handle different input shapes
        if (input.dim() == 0) {
            // Scalar tensor - reshape to 1D
            output = model->forward(input.unsqueeze(0));
        } else if (input.dim() == 1) {
            // 1D tensor - might need reshaping depending on expected input
            output = model->forward(input.unsqueeze(0));
        } else {
            // Multi-dimensional tensor
            output = model->forward(input);
        }
        
        // Test some operations on the output
        if (!output.sizes().empty()) {
            torch::Tensor sum = output.sum();
            torch::Tensor mean = output.mean();
        }
        
        // Test empty Sequential
        if (offset < Size && Data[offset] % 10 == 0) {
            torch::nn::Sequential empty_model;
            try {
                torch::Tensor empty_output = empty_model->forward(input);
            } catch (const std::exception &e) {
                // Expected exception for empty Sequential
            }
        }
        
        // Test nested Sequential using custom wrapper
        if (offset + 1 < Size && Data[offset] % 5 == 0) {
            CustomSequential nested_model;
            // Create a simple linear layer instead of nesting Sequential
            int64_t in_features = input.dim() > 0 ? input.size(-1) : 1;
            nested_model.push_back(torch::nn::Linear(in_features, 5));
            nested_model.push_back(torch::nn::ReLU());
            torch::Tensor nested_output = nested_model.forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}