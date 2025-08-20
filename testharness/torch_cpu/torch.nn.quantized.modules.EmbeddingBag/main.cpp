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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create indices tensor
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are integers
        if (indices.scalar_type() != torch::kInt && indices.scalar_type() != torch::kLong) {
            indices = indices.to(torch::kLong);
        }
        
        // Create offsets tensor if we have enough data
        if (offset < Size) {
            torch::Tensor offsets = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure offsets are integers
            if (offsets.scalar_type() != torch::kInt && offsets.scalar_type() != torch::kLong) {
                offsets = offsets.to(torch::kLong);
            }
            
            // Get embedding dimension and num_embeddings
            int64_t embedding_dim = 4;
            int64_t num_embeddings = 10;
            
            if (offset + 2 < Size) {
                // Use some bytes from the input to determine embedding_dim and num_embeddings
                embedding_dim = (Data[offset] % 8) + 1;  // 1-8
                num_embeddings = (Data[offset + 1] % 16) + 1;  // 1-16
                offset += 2;
            }
            
            // Create per_sample_weights if we have enough data
            torch::Tensor per_sample_weights;
            bool use_per_sample_weights = false;
            
            if (offset < Size && Data[offset++] % 2 == 0) {
                if (offset < Size) {
                    per_sample_weights = fuzzer_utils::createTensor(Data, Size, offset);
                    if (per_sample_weights.scalar_type() != torch::kFloat) {
                        per_sample_weights = per_sample_weights.to(torch::kFloat);
                    }
                    use_per_sample_weights = true;
                }
            }
            
            // Create embedding bag mode
            torch::nn::EmbeddingBagMode mode = torch::kSum;
            if (offset < Size) {
                int mode_val = Data[offset++] % 3;
                if (mode_val == 0) mode = torch::kSum;
                else if (mode_val == 1) mode = torch::kMean;
                else mode = torch::kMax;
            }
            
            // Create embedding bag options
            torch::nn::EmbeddingBagOptions options(num_embeddings, embedding_dim);
            options.mode(mode);
            options.sparse(offset < Size && Data[offset++] % 2 == 0);
            options.include_last_offset(offset < Size && Data[offset++] % 2 == 0);
            options.scale_grad_by_freq(offset < Size && Data[offset++] % 2 == 0);
            
            // Create embedding bag
            torch::nn::EmbeddingBag embeddingBag(options);
            
            // Forward pass
            torch::Tensor output;
            if (use_per_sample_weights) {
                output = embeddingBag->forward(indices, offsets, per_sample_weights);
            } else {
                output = embeddingBag->forward(indices, offsets);
            }
            
            // Try different mode
            if (offset < Size) {
                int mode_val = Data[offset++] % 3;
                torch::nn::EmbeddingBagMode new_mode = torch::kSum;
                if (mode_val == 0) new_mode = torch::kSum;
                else if (mode_val == 1) new_mode = torch::kMean;
                else new_mode = torch::kMax;
                
                embeddingBag->options.mode(new_mode);
                
                if (use_per_sample_weights) {
                    output = embeddingBag->forward(indices, offsets, per_sample_weights);
                } else {
                    output = embeddingBag->forward(indices, offsets);
                }
            }
            
            // Try with different parameters
            if (offset + 1 < Size) {
                offset += 2;
                
                torch::nn::EmbeddingBagOptions options2(num_embeddings, embedding_dim);
                options2.mode(mode);
                
                torch::nn::EmbeddingBag embeddingBag2(options2);
                
                if (use_per_sample_weights) {
                    output = embeddingBag2->forward(indices, offsets, per_sample_weights);
                } else {
                    output = embeddingBag2->forward(indices, offsets);
                }
            }
        } else {
            // If we don't have enough data for offsets, try with just indices
            // In this case, each sequence has length 1
            
            // Ensure indices are integers
            if (indices.scalar_type() != torch::kInt && indices.scalar_type() != torch::kLong) {
                indices = indices.to(torch::kLong);
            }
            
            int64_t embedding_dim = 4;
            int64_t num_embeddings = 10;
            
            // Create embedding bag options
            torch::nn::EmbeddingBagOptions options(num_embeddings, embedding_dim);
            options.mode(torch::kSum);
            
            // Create embedding bag
            torch::nn::EmbeddingBag embeddingBag(options);
            
            // Forward pass
            torch::Tensor output = embeddingBag->forward(indices);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}