#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters from fuzz data first
        uint8_t layer_selector = Data[offset++] % 6;
        uint8_t out_features_raw = Data[offset++];
        uint8_t mode_selector = Data[offset++];
        uint8_t extra_test = Data[offset++];
        
        int64_t out_features = (out_features_raw % 32) + 1;
        
        // Create input tensor from remaining data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is at least 2D for batch processing with Linear
        // Linear expects (*, in_features) where * is any number of batch dimensions
        if (input.numel() == 0) {
            input = torch::randn({1, 4});
        } else if (input.dim() == 0) {
            input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 1) {
            input = input.unsqueeze(0);
        }
        
        // Get in_features from the last dimension
        int64_t in_features = input.size(-1);
        if (in_features <= 0) {
            in_features = 1;
            input = torch::randn({1, 1});
        }
        
        // Ensure float type for neural network operations
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create a Sequential model with various layer combinations
        torch::nn::Sequential model;
        
        // First Linear layer
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
                model->push_back(torch::nn::LeakyReLU());
                break;
            case 5:
                model->push_back(torch::nn::Softmax(/*dim=*/-1));
                break;
        }
        
        // Add final Linear layer
        int64_t final_out = (extra_test % 8) + 1;
        model->push_back(torch::nn::Linear(out_features, final_out));
        
        // Set training mode
        if (mode_selector % 2 == 0) {
            model->eval();
        } else {
            model->train();
        }
        
        // Forward pass
        torch::Tensor output = model->forward(input);
        
        // Verify output shape and perform operations
        if (output.numel() > 0) {
            torch::Tensor sum_val = output.sum();
            torch::Tensor mean_val = output.mean();
            (void)sum_val;
            (void)mean_val;
        }
        
        // Test model parameter access
        auto params = model->parameters();
        for (const auto& p : params) {
            auto grad_fn = p.requires_grad();
            (void)grad_fn;
        }
        
        // Test named_parameters
        auto named_params = model->named_parameters();
        
        // Test modules() iteration
        auto modules = model->modules();
        
        // Test clone
        if (extra_test % 7 == 0) {
            auto cloned = model->clone();
            torch::Tensor cloned_output = cloned->as<torch::nn::Sequential>()->forward(input);
            (void)cloned_output;
        }
        
        // Test zero_grad
        if (extra_test % 5 == 0) {
            model->zero_grad();
        }
        
        // Test empty Sequential (expected to fail on forward)
        if (extra_test % 11 == 0) {
            torch::nn::Sequential empty_model;
            try {
                // Empty sequential should just pass through the input
                torch::Tensor empty_output = empty_model->forward(input);
                (void)empty_output;
            } catch (const std::exception &e) {
                // Expected - empty Sequential behavior may vary
            }
        }
        
        // Test Sequential with BatchNorm (requires specific input shape)
        if (extra_test % 13 == 0 && input.dim() >= 2) {
            torch::nn::Sequential bn_model;
            bn_model->push_back(torch::nn::Linear(in_features, 16));
            bn_model->push_back(torch::nn::BatchNorm1d(16));
            bn_model->push_back(torch::nn::ReLU());
            bn_model->push_back(torch::nn::Linear(16, 4));
            bn_model->eval();
            
            try {
                torch::Tensor bn_output = bn_model->forward(input);
                (void)bn_output;
            } catch (const std::exception &e) {
                // Shape mismatch possible with BatchNorm
            }
        }
        
        // Test Sequential with Dropout
        if (extra_test % 17 == 0) {
            torch::nn::Sequential dropout_model;
            dropout_model->push_back(torch::nn::Linear(in_features, 8));
            dropout_model->push_back(torch::nn::Dropout(0.5));
            dropout_model->push_back(torch::nn::Linear(8, 2));
            dropout_model->train();
            
            torch::Tensor drop_output = dropout_model->forward(input);
            (void)drop_output;
        }
        
        // Test to() device transfer (CPU only in this harness)
        if (extra_test % 19 == 0) {
            model->to(torch::kFloat64);
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor double_output = model->forward(double_input);
            (void)double_output;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}