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
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple model
        struct SimpleModel : torch::nn::Module {
            SimpleModel() {
                linear1 = register_module("linear1", torch::nn::Linear(10, 8));
                linear2 = register_module("linear2", torch::nn::Linear(8, 4));
                linear3 = register_module("linear3", torch::nn::Linear(4, 1));
            }
            
            torch::Tensor forward(torch::Tensor x) {
                x = torch::relu(linear1(x));
                x = torch::relu(linear2(x));
                x = linear3(x);
                return x;
            }
            
            torch::nn::Linear linear1{nullptr}, linear2{nullptr}, linear3{nullptr};
        };
        
        // Create model
        auto model = std::make_shared<SimpleModel>();
        
        // Parse input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, create a default tensor
            input = torch::randn({4, 10});
        }
        
        // Reshape input if needed to match model's expected input
        if (input.dim() == 0) {
            input = input.reshape({1, 10});
        } else if (input.dim() == 1) {
            if (input.size(0) < 10) {
                input = torch::cat({input, torch::zeros({10 - input.size(0)})});
            }
            input = input.reshape({1, input.size(0)});
        } else {
            // For tensors with dim >= 2, reshape to have 10 features in the last dimension
            std::vector<int64_t> new_shape;
            int64_t total_elements = 1;
            for (int i = 0; i < input.dim() - 1; i++) {
                new_shape.push_back(input.size(i));
                total_elements *= input.size(i);
            }
            
            if (total_elements == 0) {
                input = torch::randn({4, 10});
            } else {
                new_shape.push_back(10);
                input = input.reshape(new_shape);
            }
        }
        
        // Ensure input has the right dtype
        if (input.dtype() != torch::kFloat32) {
            input = input.to(torch::kFloat32);
        }
        
        // Since DistributedDataParallelCPU is not available in the standard PyTorch C++ API,
        // we'll simulate distributed training by just using the model directly
        // This is a simplified version for fuzzing purposes
        
        // Forward pass
        torch::Tensor output = model->forward(input);
        
        // Backward pass (if possible)
        if (output.numel() > 0 && !output.isnan().any().item<bool>() && 
            !output.isinf().any().item<bool>()) {
            try {
                auto loss = output.sum();
                loss.backward();
            } catch (const std::exception& e) {
                // Backward might fail for various reasons, that's ok for fuzzing
            }
        }
        
        // Test state_dict and load_state_dict
        try {
            auto state_dict = model->state_dict();
            model->load_state_dict(state_dict);
        } catch (const std::exception& e) {
            // This might fail, that's ok for fuzzing
        }
        
        // Test named_parameters
        for (const auto& param : model->named_parameters()) {
            // Just access the parameters to ensure they're accessible
            auto name = param.key();
            auto tensor = param.value();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}