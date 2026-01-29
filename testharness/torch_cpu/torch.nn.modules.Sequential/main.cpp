#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 2 dimensions for batch processing
        if (input.dim() == 0) {
            input = input.reshape({1, 1});
        } else if (input.dim() == 1) {
            input = input.reshape({1, input.size(0)});
        } else if (input.dim() > 2) {
            // Flatten to 2D: (batch, features)
            int64_t batch = input.size(0);
            int64_t features = input.numel() / batch;
            if (features <= 0) features = 1;
            input = input.reshape({batch, features});
        }
        
        // Get input features, ensure it's valid
        int64_t in_features = input.size(-1);
        if (in_features <= 0) {
            in_features = 1;
            input = torch::ones({1, 1});
        }
        
        // Create a Sequential module with various layers
        torch::nn::Sequential model;
        
        // Use remaining bytes to determine what layers to add
        uint8_t layer_selector = 0;
        if (offset < Size) {
            layer_selector = Data[offset++] % 6;
        }
        
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
        } else if (layer_selector == 4) {
            // Linear + BatchNorm + ReLU
            model->push_back(torch::nn::Linear(in_features, out_features));
            model->push_back(torch::nn::BatchNorm1d(out_features));
            model->push_back(torch::nn::ReLU());
        } else {
            // Single identity-like layer
            model->push_back(torch::nn::Linear(in_features, in_features));
        }
        
        // Set model to evaluation mode
        model->eval();
        
        // Forward pass with no_grad to avoid tracking gradients
        torch::Tensor output;
        {
            torch::NoGradGuard no_grad;
            output = model->forward(input);
        }
        
        // Try training mode too if we have enough data
        if (offset < Size && Data[offset] % 2 == 0) {
            offset++;
            model->train();
            torch::Tensor train_output = model->forward(input);
        }
        
        // Test named_modules iteration
        for (const auto& module : model->named_modules()) {
            (void)module.key();
            (void)module.value();
        }
        
        // Test parameters iteration
        for (const auto& param : model->parameters()) {
            (void)param.numel();
        }
        
        // Test empty input edge case if we have enough data
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                torch::Tensor empty_input = torch::empty({0, in_features});
                torch::Tensor empty_output = model->forward(empty_input);
            } catch (const std::exception&) {
                // Expected to potentially throw, just continue
            }
        }
        
        // Test nested Sequential if we have enough data
        if (offset < Size && Data[offset++] % 3 == 0) {
            torch::nn::Sequential nested;
            nested->push_back(torch::nn::Linear(in_features, 5));
            nested->push_back(torch::nn::ReLU());
            
            torch::NoGradGuard no_grad;
            torch::Tensor nested_output = nested->forward(input);
        }
        
        // Test cloning the module
        if (offset < Size && Data[offset++] % 4 == 0) {
            auto cloned = model->clone();
            torch::NoGradGuard no_grad;
            torch::Tensor cloned_output = cloned->as<torch::nn::Sequential>()->forward(input);
        }
        
        // Test zero_grad
        model->zero_grad();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}