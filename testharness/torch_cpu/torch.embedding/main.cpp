#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create weight tensor (embedding table)
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure weight tensor has at least 2 dimensions for embedding
        if (weight.dim() < 2) {
            if (weight.dim() == 0) {
                weight = weight.unsqueeze(0).unsqueeze(0);
            } else {
                weight = weight.unsqueeze(0);
            }
        }
        
        // Ensure weight is float type for embedding
        if (!weight.is_floating_point()) {
            weight = weight.to(torch::kFloat);
        }
        
        int64_t num_embeddings = weight.size(0);
        if (num_embeddings == 0) {
            return 0;  // Can't do embedding with empty table
        }
        
        // Create indices tensor
        torch::Tensor indices;
        if (offset + 4 <= Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to long type
            indices = indices.to(torch::kLong);
            
            // Clamp indices to valid range [0, num_embeddings - 1]
            indices = indices.abs().remainder(num_embeddings);
        } else {
            // Create default indices if we don't have enough data
            indices = torch::zeros({2}, torch::kLong);
        }
        
        // Extract embedding parameters from remaining data
        int64_t padding_idx = -1;  // -1 means no padding
        bool scale_grad_by_freq = false;
        bool sparse = false;
        
        if (offset + 3 <= Size) {
            // Extract padding_idx - use modulo to get valid index or -1
            int8_t raw_idx = static_cast<int8_t>(Data[offset++]);
            if (raw_idx < 0) {
                padding_idx = -1;  // No padding
            } else {
                padding_idx = raw_idx % num_embeddings;
            }
            
            // Extract scale_grad_by_freq as a boolean
            scale_grad_by_freq = (Data[offset++] % 2 == 1);
            
            // Extract sparse as a boolean
            sparse = (Data[offset++] % 2 == 1);
        }
        
        // Apply embedding operation
        torch::Tensor result = torch::embedding(
            weight,
            indices,
            padding_idx,
            scale_grad_by_freq,
            sparse
        );
        
        // Perform some operations on the result to ensure it's used
        volatile float sum_val = result.sum().item<float>();
        (void)sum_val;
        
        // Try different variants of embedding
        if (offset < Size) {
            // Try with different padding_idx
            int64_t new_padding_idx = static_cast<int64_t>(Data[offset++]) % (num_embeddings + 1) - 1;
            result = torch::embedding(weight, indices, new_padding_idx, scale_grad_by_freq, sparse);
            sum_val = result.sum().item<float>();
            
            // Try with toggled scale_grad_by_freq
            result = torch::embedding(weight, indices, padding_idx, !scale_grad_by_freq, sparse);
            sum_val = result.sum().item<float>();
            
            // Try with toggled sparse
            result = torch::embedding(weight, indices, padding_idx, scale_grad_by_freq, !sparse);
            sum_val = result.sum().item<float>();
        }
        
        // Try with empty indices
        if (offset < Size && Data[offset++] % 4 == 0) {
            torch::Tensor empty_indices = torch::empty({0}, torch::kLong);
            result = torch::embedding(weight, empty_indices, padding_idx, scale_grad_by_freq, sparse);
            sum_val = result.sum().item<float>();
        }
        
        // Try with multi-dimensional indices
        if (offset + 2 <= Size) {
            int64_t dim0 = (Data[offset++] % 4) + 1;
            int64_t dim1 = (Data[offset++] % 4) + 1;
            torch::Tensor multi_indices = torch::randint(0, num_embeddings, {dim0, dim1}, torch::kLong);
            result = torch::embedding(weight, multi_indices, padding_idx, scale_grad_by_freq, sparse);
            sum_val = result.sum().item<float>();
        }
        
        // Try with 1D weight (should work after reshape)
        if (offset < Size && weight.numel() > 1) {
            int64_t embedding_dim = weight.size(1);
            torch::Tensor reshaped_weight = weight.reshape({-1, embedding_dim});
            int64_t new_num_embeddings = reshaped_weight.size(0);
            torch::Tensor valid_indices = torch::zeros({2}, torch::kLong);
            if (new_num_embeddings > 1) {
                valid_indices = torch::tensor({0, new_num_embeddings - 1}, torch::kLong);
            }
            result = torch::embedding(reshaped_weight, valid_indices, -1, false, false);
            sum_val = result.sum().item<float>();
        }
        
        // Try with different dtypes for weight
        if (offset < Size) {
            try {
                torch::Tensor double_weight = weight.to(torch::kDouble);
                result = torch::embedding(double_weight, indices, padding_idx, scale_grad_by_freq, sparse);
                sum_val = result.sum().item<float>();
            } catch (const std::exception&) {
                // Some configurations might not support double
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