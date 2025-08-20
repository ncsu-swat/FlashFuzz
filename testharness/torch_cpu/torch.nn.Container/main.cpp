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
        
        // Create a container
        torch::nn::Sequential container;
        
        // Determine number of layers to add (1-4)
        uint8_t num_layers = (Size > 0) ? (Data[offset] % 4) + 1 : 1;
        offset++;
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input = torch::randn({2, 3});
        }
        
        // Add layers to the container
        for (uint8_t i = 0; i < num_layers && offset < Size; i++) {
            uint8_t layer_type = (offset < Size) ? Data[offset] % 5 : 0;
            offset = std::min(offset + 1, Size);
            
            int64_t in_features = input.size(input.dim() > 0 ? 0 : 0);
            int64_t out_features = (offset < Size) ? (Data[offset] % 10) + 1 : 5;
            offset = std::min(offset + 1, Size);
            
            switch (layer_type) {
                case 0:
                    container->push_back(torch::nn::Linear(in_features, out_features));
                    break;
                case 1:
                    container->push_back(torch::nn::ReLU());
                    break;
                case 2:
                    container->push_back(torch::nn::Dropout(0.5));
                    break;
                case 3:
                    container->push_back(torch::nn::Tanh());
                    break;
                case 4:
                    container->push_back(torch::nn::Sigmoid());
                    break;
                default:
                    container->push_back(torch::nn::ReLU());
            }
            
            // Update input shape for next layer
            if (layer_type == 0) { // Linear layer changes shape
                if (input.dim() > 0) {
                    std::vector<int64_t> new_shape = input.sizes().vec();
                    if (!new_shape.empty()) {
                        new_shape[0] = out_features;
                    }
                    input = torch::randn(new_shape);
                } else {
                    input = torch::randn({out_features});
                }
            }
        }
        
        // Test container operations
        
        // 1. Forward pass
        try {
            torch::Tensor output = container->forward(input);
        } catch (...) {
            // Forward might fail due to shape mismatches, which is expected
        }
        
        // 2. Test named_children
        for (const auto& child : container->named_children()) {
            auto name = child.key();
            auto module = child.value();
        }
        
        // 3. Test parameters
        for (const auto& param : container->parameters()) {
            auto param_size = param.sizes();
        }
        
        // 4. Test to method
        if (torch::cuda::is_available()) {
            try {
                container->to(torch::kCUDA);
            } catch (...) {
                // CUDA operations might fail
            }
        }
        
        // 5. Test empty container
        torch::nn::Sequential empty_container;
        try {
            if (input.dim() > 0) {
                torch::Tensor empty_output = empty_container->forward(input);
            }
        } catch (...) {
            // Empty container forward might fail
        }
        
        // 6. Test adding modules with names
        torch::nn::ModuleDict named_container;
        try {
            named_container["layer1"] = torch::nn::Linear(10, 5);
            named_container["layer2"] = torch::nn::ReLU();
        } catch (...) {
            // Insertion might fail
        }
        
        // 7. Test container with different module types
        torch::nn::Sequential mixed_container(
            torch::nn::Linear(10, 5),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.2)
        );
        
        // 8. Test container methods
        container->eval();
        container->train();
        container->zero_grad();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}