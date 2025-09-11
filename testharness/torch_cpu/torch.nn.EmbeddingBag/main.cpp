#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input indices tensor
        torch::Tensor indices;
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create input weights tensor
        torch::Tensor weights;
        bool has_weights = false;
        if (offset < Size && Data[offset] % 2 == 0) {
            offset++;
            weights = fuzzer_utils::createTensor(Data, Size, offset);
            has_weights = true;
        } else if (offset < Size) {
            offset++;
        }
        
        // Create input offsets tensor
        torch::Tensor offsets;
        bool has_offsets = false;
        if (offset < Size && Data[offset] % 2 == 0) {
            offset++;
            offsets = fuzzer_utils::createTensor(Data, Size, offset);
            has_offsets = true;
        } else if (offset < Size) {
            offset++;
        }
        
        // Parse EmbeddingBag parameters
        int64_t num_embeddings = 10;
        int64_t embedding_dim = 3;
        
        if (offset + 2 < Size) {
            num_embeddings = static_cast<int64_t>(Data[offset]) + 1; // Ensure at least 1
            embedding_dim = static_cast<int64_t>(Data[offset + 1]) + 1; // Ensure at least 1
            offset += 2;
        }
        
        // Parse mode
        torch::nn::EmbeddingBagMode mode = torch::kSum;
        if (offset < Size) {
            int mode_val = Data[offset] % 3;
            if (mode_val == 0) {
                mode = torch::kSum;
            } else if (mode_val == 1) {
                mode = torch::kMean;
            } else {
                mode = torch::kMax;
            }
            offset++;
        }
        
        // Parse sparse flag
        bool sparse = false;
        if (offset < Size) {
            sparse = (Data[offset] % 2 == 0);
            offset++;
        }
        
        // Parse scale_grad_by_freq flag
        bool scale_grad_by_freq = false;
        if (offset < Size) {
            scale_grad_by_freq = (Data[offset] % 2 == 0);
            offset++;
        }
        
        // Parse padding_idx
        int64_t padding_idx = -1;
        if (offset < Size) {
            if (Data[offset] % 3 == 0) {
                padding_idx = -1; // No padding
            } else if (Data[offset] % 3 == 1) {
                padding_idx = Data[offset] % num_embeddings; // Valid padding index
            } else {
                padding_idx = num_embeddings + 10; // Out of range padding index
            }
            offset++;
        }
        
        // Create EmbeddingBag module
        torch::nn::EmbeddingBagOptions options = torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
            .mode(mode)
            .sparse(sparse)
            .scale_grad_by_freq(scale_grad_by_freq)
            .padding_idx(padding_idx);
        
        auto embeddingBag = torch::nn::EmbeddingBag(options);
        
        // Forward pass
        torch::Tensor output;
        if (has_offsets && has_weights) {
            output = embeddingBag->forward(indices, offsets, weights);
        } else if (has_offsets) {
            output = embeddingBag->forward(indices, offsets);
        } else if (has_weights) {
            output = embeddingBag->forward(indices, torch::Tensor(), weights);
        } else {
            output = embeddingBag->forward(indices);
        }
        
        // Test backward pass
        if (output.requires_grad()) {
            auto grad_output = torch::ones_like(output);
            output.backward(grad_output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
