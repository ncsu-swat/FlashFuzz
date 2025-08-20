#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a module with a linear layer
        torch::nn::Linear linear = nullptr;
        
        // Parse input size and output size for the linear layer
        uint16_t in_features = 0;
        uint16_t out_features = 0;
        
        if (offset + sizeof(uint16_t) <= Size) {
            std::memcpy(&in_features, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
        }
        
        if (offset + sizeof(uint16_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
        }
        
        // Ensure we have at least 1 feature in each dimension
        in_features = (in_features % 100) + 1;
        out_features = (out_features % 100) + 1;
        
        // Create the linear layer
        linear = torch::nn::Linear(in_features, out_features);
        
        // Create a sequential module to hold our linear layer
        torch::nn::Sequential model(linear);
        
        // Apply spectral norm to the model
        torch::nn::utils::spectral_norm(model);
        
        // Create a random input tensor to test the model
        torch::Tensor input;
        if (offset < Size) {
            try {
                input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Reshape input if necessary to match the expected input shape
                if (input.dim() == 0) {
                    input = input.reshape({1, in_features});
                } else if (input.dim() == 1) {
                    if (input.size(0) != in_features) {
                        input = input.reshape({1, -1});
                        if (input.size(1) != in_features) {
                            input = torch::ones({1, in_features});
                        }
                    } else {
                        input = input.reshape({1, in_features});
                    }
                } else {
                    // For tensors with dim > 1, reshape the last dimension to match in_features
                    std::vector<int64_t> new_shape = input.sizes().vec();
                    if (new_shape.size() >= 2) {
                        new_shape[new_shape.size() - 1] = in_features;
                        input = input.reshape(new_shape);
                    } else {
                        input = torch::ones({1, in_features});
                    }
                }
                
                // Forward pass to ensure spectral norm is applied
                torch::Tensor output = model->forward(input);
            } catch (const std::exception& e) {
                // If tensor creation fails, use a default tensor
                input = torch::ones({1, in_features});
            }
        } else {
            input = torch::ones({1, in_features});
        }
        
        // Test the remove_spectral_norm function
        torch::nn::utils::remove_spectral_norm(model);
        
        // Test the model after removing spectral norm
        torch::Tensor output_after_removal = model->forward(input);
        
        // Try with different module types
        if (Size > offset + 1) {
            uint8_t module_type = Data[offset++];
            
            // Create different module types based on the input data
            torch::nn::AnyModule any_module;
            
            switch (module_type % 5) {
                case 0: {
                    // Conv1d
                    auto conv1d = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_features, out_features, 3).padding(1));
                    auto seq = torch::nn::Sequential(conv1d);
                    torch::nn::utils::spectral_norm(seq);
                    torch::nn::utils::remove_spectral_norm(seq);
                    break;
                }
                case 1: {
                    // Conv2d
                    auto conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_features, out_features, 3).padding(1));
                    auto seq = torch::nn::Sequential(conv2d);
                    torch::nn::utils::spectral_norm(seq);
                    torch::nn::utils::remove_spectral_norm(seq);
                    break;
                }
                case 2: {
                    // Conv3d
                    auto conv3d = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_features, out_features, 3).padding(1));
                    auto seq = torch::nn::Sequential(conv3d);
                    torch::nn::utils::spectral_norm(seq);
                    torch::nn::utils::remove_spectral_norm(seq);
                    break;
                }
                case 3: {
                    // Linear
                    auto linear_module = torch::nn::Linear(in_features, out_features);
                    auto seq = torch::nn::Sequential(linear_module);
                    torch::nn::utils::spectral_norm(seq);
                    torch::nn::utils::remove_spectral_norm(seq);
                    break;
                }
                case 4: {
                    // Embedding
                    auto embedding = torch::nn::Embedding(in_features, out_features);
                    auto seq = torch::nn::Sequential(embedding);
                    torch::nn::utils::spectral_norm(seq);
                    torch::nn::utils::remove_spectral_norm(seq);
                    break;
                }
            }
        }
        
        // Test edge cases
        if (Size > offset) {
            try {
                // Try to remove spectral norm from a module that doesn't have it
                auto plain_linear = torch::nn::Linear(in_features, out_features);
                auto plain_seq = torch::nn::Sequential(plain_linear);
                torch::nn::utils::remove_spectral_norm(plain_seq);
            } catch (const std::exception& e) {
                // Expected exception, continue
            }
            
            // Try to remove spectral norm with custom name
            if (Size > offset + 1) {
                auto custom_linear = torch::nn::Linear(in_features, out_features);
                auto custom_seq = torch::nn::Sequential(custom_linear);
                
                // Get a custom name from the input data
                std::string custom_name = "weight_orig";
                if (Data[offset] % 2 == 0) {
                    custom_name = "custom_spectral_norm";
                }
                
                // Apply spectral norm with custom name
                torch::nn::utils::spectral_norm(custom_seq, custom_name);
                
                // Remove spectral norm with the same custom name
                torch::nn::utils::remove_spectral_norm(custom_seq, custom_name);
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