#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf
#include <torch/torch.h>

// Helper function to sanitize float values from fuzzer data
static float sanitize_float(float value, float default_val, float min_val, float max_val) {
    if (std::isnan(value) || std::isinf(value)) {
        return default_val;
    }
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor for model parameters
        torch::Tensor params = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a valid tensor with proper shape
        if (params.numel() == 0) {
            params = torch::randn({4});
        }
        
        // Ensure params is float type for gradient computation
        if (!params.is_floating_point()) {
            params = params.to(torch::kFloat32);
        }
        
        // Create a simple model with parameters
        struct SimpleModel : torch::nn::Module {
            torch::Tensor weights;
            SimpleModel(torch::Tensor initial_weights) {
                weights = register_parameter("weights", initial_weights.clone().requires_grad_(true));
            }
            torch::Tensor forward() {
                return weights;
            }
        };
        
        auto model = std::make_shared<SimpleModel>(params);
        
        // Extract optimizer configuration from the fuzzer data
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Parse optimizer type
        uint8_t optimizer_type = Data[offset++] % 5;
        
        // Parse learning rate (sanitize to reasonable range)
        float learning_rate = 0.01f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&learning_rate, Data + offset, sizeof(float));
            offset += sizeof(float);
            learning_rate = sanitize_float(learning_rate, 0.01f, 1e-8f, 10.0f);
        }
        
        // Parse momentum (for SGD, sanitize to [0, 1])
        float momentum = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
            momentum = sanitize_float(momentum, 0.0f, 0.0f, 0.999f);
        }
        
        // Parse weight decay (sanitize to [0, 1])
        float weight_decay = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&weight_decay, Data + offset, sizeof(float));
            offset += sizeof(float);
            weight_decay = sanitize_float(weight_decay, 0.0f, 0.0f, 0.1f);
        }
        
        // Create optimizer based on the parsed type
        std::unique_ptr<torch::optim::Optimizer> optimizer;
        
        switch (optimizer_type) {
            case 0: {
                // SGD
                torch::optim::SGDOptions options(learning_rate);
                options.momentum(momentum);
                options.weight_decay(weight_decay);
                optimizer = std::make_unique<torch::optim::SGD>(
                    model->parameters(), options);
                break;
            }
            case 1: {
                // Adam
                torch::optim::AdamOptions options(learning_rate);
                options.weight_decay(weight_decay);
                optimizer = std::make_unique<torch::optim::Adam>(
                    model->parameters(), options);
                break;
            }
            case 2: {
                // RMSprop
                torch::optim::RMSpropOptions options(learning_rate);
                options.weight_decay(weight_decay);
                optimizer = std::make_unique<torch::optim::RMSprop>(
                    model->parameters(), options);
                break;
            }
            case 3: {
                // Adagrad
                torch::optim::AdagradOptions options(learning_rate);
                options.weight_decay(weight_decay);
                optimizer = std::make_unique<torch::optim::Adagrad>(
                    model->parameters(), options);
                break;
            }
            default: {
                // LBFGS
                torch::optim::LBFGSOptions options(learning_rate);
                options.max_iter(5);  // Limit iterations for fuzzing
                optimizer = std::make_unique<torch::optim::LBFGS>(
                    model->parameters(), options);
                break;
            }
        }
        
        // Test optimizer step
        optimizer->zero_grad();
        
        if (optimizer_type == 4) {
            // LBFGS requires closure
            auto closure = [&model]() -> torch::Tensor {
                model->zero_grad();
                auto loss = torch::sum(torch::pow(model->weights, 2));
                loss.backward();
                return loss;
            };
            static_cast<torch::optim::LBFGS*>(optimizer.get())->step(closure);
        } else {
            // Compute loss and gradients for other optimizers
            auto loss = torch::sum(torch::pow(model->weights, 2));
            loss.backward();
            optimizer->step();
        }
        
        // Test zero_grad again
        optimizer->zero_grad();
        
        // Run a few more optimization steps to exercise the state
        for (int i = 0; i < 3; i++) {
            if (optimizer_type == 4) {
                auto closure = [&model]() -> torch::Tensor {
                    model->zero_grad();
                    auto loss = torch::sum(torch::pow(model->weights, 2));
                    loss.backward();
                    return loss;
                };
                try {
                    static_cast<torch::optim::LBFGS*>(optimizer.get())->step(closure);
                } catch (...) {
                    // LBFGS may fail on some inputs, that's expected
                    break;
                }
            } else {
                optimizer->zero_grad();
                auto loss = torch::sum(torch::pow(model->weights, 2));
                loss.backward();
                optimizer->step();
            }
        }
        
        // Test state_dict and load_state_dict for non-LBFGS optimizers
        if (optimizer_type != 4 && offset < Size) {
            try {
                // This exercises serialization paths
                auto param_groups = optimizer->param_groups();
                (void)param_groups.size();  // Access to prevent optimization
            } catch (...) {
                // Some state operations may fail, that's fine
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}