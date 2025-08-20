#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create embedding weight matrix
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure weight has at least 2 dimensions for embedding
        if (weight.dim() < 2) {
            if (weight.dim() == 0) {
                weight = weight.unsqueeze(0).unsqueeze(0);
            } else {
                weight = weight.unsqueeze(0);
            }
        }
        
        // Extract parameters for embedding_renorm_
        // Get max_norm parameter (should be positive)
        float max_norm = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure max_norm is positive (can be any positive value)
            max_norm = std::abs(max_norm) + 1e-6f;
        } else {
            max_norm = 1.0f; // Default value
        }
        
        // Get norm_type parameter
        float norm_type = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(float));
            offset += sizeof(float);
        } else {
            norm_type = 2.0f; // Default to L2 norm
        }
        
        // Create indices tensor for which embeddings to renormalize
        std::vector<int64_t> indices_vec;
        int num_indices = 0;
        
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&num_indices, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // Ensure num_indices is reasonable
            num_indices = std::abs(num_indices) % 10 + 1;
            
            // Generate indices within the valid range
            int64_t num_embeddings = weight.size(0);
            
            for (int i = 0; i < num_indices && offset + sizeof(int64_t) <= Size; i++) {
                int64_t idx = 0;
                std::memcpy(&idx, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Ensure index is within bounds
                idx = std::abs(idx) % std::max<int64_t>(1, num_embeddings);
                indices_vec.push_back(idx);
            }
        }
        
        // If we couldn't extract any indices, create a default one
        if (indices_vec.empty()) {
            indices_vec.push_back(0);
        }
        
        // Create indices tensor
        torch::Tensor indices = torch::tensor(indices_vec, torch::kInt64);
        
        // Apply embedding_renorm_
        torch::embedding_renorm_(weight, indices, max_norm, norm_type);
        
        // Try with different parameters to increase coverage
        if (offset + sizeof(float) <= Size) {
            float new_max_norm;
            std::memcpy(&new_max_norm, Data + offset, sizeof(float));
            offset += sizeof(float);
            new_max_norm = std::abs(new_max_norm) + 1e-6f;
            
            // Try with different norm_type
            float new_norm_type = 1.0f; // L1 norm
            if (norm_type == 1.0f) {
                new_norm_type = 2.0f; // L2 norm
            }
            
            // Apply with different parameters
            torch::embedding_renorm_(weight, indices, new_max_norm, new_norm_type);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}