#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// Custom Sequential wrapper to avoid template issues
class CustomSequential : public torch::nn::Module {
public:
    CustomSequential() {
        register_module("sequential", sequential_);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        return sequential_->forward(x);
    }
    
    void push_back(torch::nn::AnyModule module) {
        sequential_->push_back(module);
    }
    
    void train(bool on = true) override {
        sequential_->train(on);
    }
    
    void eval() override {
        sequential_->eval();
    }
    
private:
    torch::nn::Sequential sequential_;
};

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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a Sequential module with various layers
        torch::nn::Sequential model;
        
        // Use remaining bytes to determine what layers to add
        if (offset < Size) {
            uint8_t layer_selector = Data[offset++] % 5;
            
            // Add a Linear layer
            int64_t in_features = input.dim() > 0 ? input.size(-1) : 1;
            int64_t out_features = 10;
            
            if (layer_selector == 0) {
                // Simple linear layer
                model->push_back(torch::nn::Linear(in_features, out_features));
            } else if (layer_selector == 1) {
                // Linear + ReLU
                model->push_back(torch::nn::Linear(in_features, out_features));
                model->push_back(torch::nn::ReLU());
            } else if (layer_selector == 2) {
                // Linear + Dropout + ReLU
                model->push_back(torch::nn::Linear(in_features, out_features));
                model->push_back(torch::nn::Dropout(0.5));
                model->push_back(torch::nn::ReLU());
            } else if (layer_selector == 3) {
                // Multiple linear layers
                model->push_back(torch::nn::Linear(in_features, out_features));
                model->push_back(torch::nn::ReLU());
                model->push_back(torch::nn::Linear(out_features, 5));
            } else {
                // Empty sequential (edge case)
            }
        } else {
            // Default case: just add a simple linear layer
            int64_t in_features = input.dim() > 0 ? input.size(-1) : 1;
            model->push_back(torch::nn::Linear(in_features, 10));
        }
        
        // Try to reshape input if needed for the model
        if (input.dim() == 0) {
            // Scalar tensor needs reshaping to have at least one dimension
            input = input.reshape({1, 1});
        } else if (input.dim() == 1) {
            // 1D tensor needs batch dimension
            input = input.reshape({1, input.size(0)});
        }
        
        // Apply the model to the input
        torch::Tensor output;
        
        // Set model to evaluation mode
        model->eval();
        
        // Forward pass with no_grad to avoid tracking gradients
        {
            torch::NoGradGuard no_grad;
            output = model->forward(input);
        }
        
        // Try training mode too if we have enough data
        if (offset < Size) {
            model->train();
            torch::Tensor train_output = model->forward(input);
        }
        
        // Test empty input edge case if we have enough data
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                torch::Tensor empty_input = torch::empty({0, input.size(-1)});
                torch::Tensor empty_output = model->forward(empty_input);
            } catch (const std::exception&) {
                // Expected to potentially throw, just continue
            }
        }
        
        // Test nested Sequential if we have enough data
        if (offset < Size && Data[offset++] % 2 == 0) {
            CustomSequential nested;
            nested.push_back(torch::nn::Linear(input.size(-1), 5));
            nested.push_back(torch::nn::ReLU());
            
            torch::Tensor nested_output = nested.forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
