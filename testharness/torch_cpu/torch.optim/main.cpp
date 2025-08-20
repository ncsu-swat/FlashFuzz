#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for model parameters
        torch::Tensor params = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple model with parameters
        struct SimpleModel : torch::nn::Module {
            torch::Tensor weights;
            SimpleModel(torch::Tensor initial_weights) {
                weights = register_parameter("weights", initial_weights);
            }
            torch::Tensor forward() {
                return weights;
            }
        };
        
        auto model = std::make_shared<SimpleModel>(params.clone().requires_grad_(true));
        
        // Extract optimizer configuration from the fuzzer data
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Parse optimizer type
        uint8_t optimizer_type = Data[offset++] % 5;
        
        // Parse learning rate
        float learning_rate = 0.01f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&learning_rate, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Parse momentum (for SGD)
        float momentum = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Parse weight decay
        float weight_decay = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&weight_decay, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Create optimizer based on the parsed type
        std::shared_ptr<torch::optim::Optimizer> optimizer;
        
        switch (optimizer_type) {
            case 0: {
                // SGD
                torch::optim::SGDOptions options(learning_rate);
                options.momentum(momentum);
                options.weight_decay(weight_decay);
                optimizer = std::make_shared<torch::optim::SGD>(
                    model->parameters(), options);
                break;
            }
            case 1: {
                // Adam
                torch::optim::AdamOptions options(learning_rate);
                options.weight_decay(weight_decay);
                optimizer = std::make_shared<torch::optim::Adam>(
                    model->parameters(), options);
                break;
            }
            case 2: {
                // RMSprop
                torch::optim::RMSpropOptions options(learning_rate);
                options.weight_decay(weight_decay);
                optimizer = std::make_shared<torch::optim::RMSprop>(
                    model->parameters(), options);
                break;
            }
            case 3: {
                // Adagrad
                torch::optim::AdagradOptions options(learning_rate);
                options.weight_decay(weight_decay);
                optimizer = std::make_shared<torch::optim::Adagrad>(
                    model->parameters(), options);
                break;
            }
            default: {
                // LBFGS
                torch::optim::LBFGSOptions options(learning_rate);
                optimizer = std::make_shared<torch::optim::LBFGS>(
                    model->parameters(), options);
                break;
            }
        }
        
        // Create a simple loss function
        auto loss_fn = [&]() {
            return torch::sum(torch::pow(model->weights, 2));
        };
        
        // Test optimizer step
        optimizer->zero_grad();
        
        // Compute loss and gradients
        auto loss = loss_fn();
        loss.backward();
        
        // Step the optimizer
        if (optimizer_type == 4) {  // LBFGS requires closure
            static_cast<torch::optim::LBFGS&>(*optimizer).step(loss_fn);
        } else {
            optimizer->step();
        }
        
        // Test additional optimizer methods
        optimizer->zero_grad();
        
        // Test adding parameter groups
        if (offset + 1 < Size) {
            uint8_t add_param_group = Data[offset++];
            if (add_param_group % 2 == 0) {
                // Create another tensor for a new parameter group
                torch::Tensor extra_params = fuzzer_utils::createTensor(Data, Size, offset);
                extra_params = extra_params.clone().requires_grad_(true);
                
                // Add parameter group with different learning rate
                std::vector<torch::Tensor> param_vec = {extra_params};
                
                if (optimizer_type == 0) {  // SGD
                    torch::optim::SGDOptions options(learning_rate * 2);
                    options.momentum(momentum);
                    options.weight_decay(weight_decay);
                    torch::optim::OptimizerParamGroup param_group(
                        param_vec, std::make_unique<torch::optim::SGDOptions>(options));
                    static_cast<torch::optim::SGD&>(*optimizer).add_param_group(param_group);
                } else if (optimizer_type == 1) {  // Adam
                    torch::optim::AdamOptions options(learning_rate * 2);
                    options.weight_decay(weight_decay);
                    torch::optim::OptimizerParamGroup param_group(
                        param_vec, std::make_unique<torch::optim::AdamOptions>(options));
                    static_cast<torch::optim::Adam&>(*optimizer).add_param_group(param_group);
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}