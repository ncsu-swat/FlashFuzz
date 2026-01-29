#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        if (Size < 5) {
            return 0;
        }
        
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
        
        // Ensure we have at least 1 feature in each dimension (limit to reasonable sizes)
        in_features = (in_features % 64) + 1;
        out_features = (out_features % 64) + 1;
        
        uint8_t module_type = 0;
        if (offset < Size) {
            module_type = Data[offset++];
        }
        
        // Parse optional parameters for spectral_norm
        int64_t n_power_iterations = 1;
        double eps = 1e-12;
        int64_t dim = 0;
        
        if (offset + 1 <= Size) {
            n_power_iterations = (Data[offset++] % 5) + 1; // 1-5 iterations
        }
        
        // Test different module types
        switch (module_type % 4) {
            case 0: {
                // Linear module
                auto linear = torch::nn::Linear(in_features, out_features);
                
                // Apply spectral norm
                torch::nn::utils::spectral_norm(linear, "weight", n_power_iterations, eps, dim);
                
                // Create input and do forward pass to exercise the normalized weights
                auto input = torch::randn({1, in_features});
                try {
                    auto output = linear->forward(input);
                } catch (...) {
                    // Shape mismatch, ignore
                }
                
                // Remove spectral norm
                torch::nn::utils::remove_spectral_norm(linear, "weight");
                
                // Forward pass after removal
                try {
                    auto output_after = linear->forward(input);
                } catch (...) {
                    // Ignore
                }
                break;
            }
            case 1: {
                // Conv1d module
                int64_t kernel_size = 3;
                auto conv1d = torch::nn::Conv1d(
                    torch::nn::Conv1dOptions(in_features, out_features, kernel_size).padding(1));
                
                // Apply spectral norm (dim=0 is default for Conv)
                torch::nn::utils::spectral_norm(conv1d, "weight", n_power_iterations, eps, 0);
                
                // Create input and do forward pass
                auto input = torch::randn({1, in_features, 10});
                try {
                    auto output = conv1d->forward(input);
                } catch (...) {
                    // Ignore shape errors
                }
                
                // Remove spectral norm
                torch::nn::utils::remove_spectral_norm(conv1d, "weight");
                
                // Forward pass after removal
                try {
                    auto output_after = conv1d->forward(input);
                } catch (...) {
                    // Ignore
                }
                break;
            }
            case 2: {
                // Conv2d module
                int64_t kernel_size = 3;
                auto conv2d = torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_features, out_features, kernel_size).padding(1));
                
                // Apply spectral norm
                torch::nn::utils::spectral_norm(conv2d, "weight", n_power_iterations, eps, 0);
                
                // Create input and do forward pass
                auto input = torch::randn({1, in_features, 8, 8});
                try {
                    auto output = conv2d->forward(input);
                } catch (...) {
                    // Ignore shape errors
                }
                
                // Remove spectral norm
                torch::nn::utils::remove_spectral_norm(conv2d, "weight");
                
                // Forward pass after removal
                try {
                    auto output_after = conv2d->forward(input);
                } catch (...) {
                    // Ignore
                }
                break;
            }
            case 3: {
                // Embedding module
                int64_t num_embeddings = in_features;
                int64_t embedding_dim = out_features;
                auto embedding = torch::nn::Embedding(num_embeddings, embedding_dim);
                
                // Apply spectral norm (dim=1 is typical for embedding)
                torch::nn::utils::spectral_norm(embedding, "weight", n_power_iterations, eps, 1);
                
                // Create input and do forward pass
                auto input = torch::randint(0, num_embeddings, {1, 5});
                try {
                    auto output = embedding->forward(input);
                } catch (...) {
                    // Ignore
                }
                
                // Remove spectral norm
                torch::nn::utils::remove_spectral_norm(embedding, "weight");
                
                // Forward pass after removal
                try {
                    auto output_after = embedding->forward(input);
                } catch (...) {
                    // Ignore
                }
                break;
            }
        }
        
        // Test edge case: try to remove spectral norm from module that doesn't have it
        if (offset < Size && (Data[offset] % 4 == 0)) {
            auto plain_linear = torch::nn::Linear(in_features, out_features);
            try {
                // This should throw an exception
                torch::nn::utils::remove_spectral_norm(plain_linear, "weight");
            } catch (const std::exception& e) {
                // Expected: module doesn't have spectral norm applied
            }
        }
        
        // Test applying and removing spectral norm multiple times
        if (offset < Size && (Data[offset] % 3 == 0)) {
            auto linear = torch::nn::Linear(in_features, out_features);
            
            // Apply spectral norm
            torch::nn::utils::spectral_norm(linear, "weight", 1, 1e-12, 0);
            
            // Remove it
            torch::nn::utils::remove_spectral_norm(linear, "weight");
            
            // Apply again
            torch::nn::utils::spectral_norm(linear, "weight", 2, 1e-10, 0);
            
            // Do a forward pass
            auto input = torch::randn({1, in_features});
            try {
                auto output = linear->forward(input);
            } catch (...) {
                // Ignore
            }
            
            // Remove again
            torch::nn::utils::remove_spectral_norm(linear, "weight");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}