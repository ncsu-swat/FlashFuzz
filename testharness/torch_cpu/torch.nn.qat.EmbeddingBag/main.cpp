#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors for EmbeddingBag
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are integers
        if (indices.scalar_type() != torch::kInt && indices.scalar_type() != torch::kLong) {
            indices = indices.to(torch::kLong);
        }
        
        // Create offsets tensor
        torch::Tensor offsets;
        if (offset < Size) {
            offsets = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure offsets are integers
            if (offsets.scalar_type() != torch::kInt && offsets.scalar_type() != torch::kLong) {
                offsets = offsets.to(torch::kLong);
            }
        } else {
            // Default offsets if we don't have enough data
            offsets = torch::zeros({1}, torch::kLong);
        }
        
        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default weight if we don't have enough data
            weight = torch::ones({10, 5}, torch::dtype(torch::kFloat));
        }
        
        // Extract parameters for EmbeddingBag
        int64_t num_embeddings = 10;
        int64_t embedding_dim = 5;
        
        if (!weight.sizes().empty()) {
            if (weight.dim() >= 2) {
                num_embeddings = weight.size(0);
                embedding_dim = weight.size(1);
            } else if (weight.dim() == 1) {
                num_embeddings = weight.size(0);
                embedding_dim = 1;
            }
        }
        
        // Extract additional parameters from the input data
        bool sparse = false;
        int64_t mode = 0;
        bool include_last_offset = false;
        
        if (offset + 3 <= Size) {
            sparse = Data[offset++] % 2 == 0;
            mode = Data[offset++] % 3;  // 0: sum, 1: mean, 2: max
            include_last_offset = Data[offset++] % 2 == 0;
        }
        
        // Convert mode to EmbeddingBagMode
        torch::nn::EmbeddingBagMode embedding_mode;
        switch (mode) {
            case 0:
                embedding_mode = torch::kSum;
                break;
            case 1:
                embedding_mode = torch::kMean;
                break;
            case 2:
                embedding_mode = torch::kMax;
                break;
            default:
                embedding_mode = torch::kSum;
                break;
        }
        
        // Create EmbeddingBag module (using regular EmbeddingBag since QAT version is not available)
        torch::nn::EmbeddingBagOptions options = 
            torch::nn::EmbeddingBagOptions(num_embeddings, embedding_dim)
                .sparse(sparse)
                .mode(embedding_mode)
                .include_last_offset(include_last_offset);
        
        auto embedding_bag = torch::nn::EmbeddingBag(options);
        
        // Set the weight
        embedding_bag->weight = weight;
        
        // Forward pass
        torch::Tensor output;
        if (offsets.numel() > 0) {
            output = embedding_bag->forward(indices, offsets);
        } else {
            output = embedding_bag->forward(indices);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}