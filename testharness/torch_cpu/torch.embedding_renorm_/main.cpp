#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Ensure weight is float type (embedding_renorm_ requires floating point)
        if (!weight.is_floating_point()) {
            weight = weight.to(torch::kFloat32);
        }
        
        // Extract parameters for embedding_renorm_
        // Get max_norm parameter (should be positive)
        float max_norm = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_norm, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure max_norm is positive and finite
            if (!std::isfinite(max_norm) || max_norm <= 0) {
                max_norm = 1.0f;
            }
        }
        
        // Get norm_type parameter
        float norm_type = 2.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure norm_type is finite and reasonable (typically 1.0 or 2.0)
            if (!std::isfinite(norm_type)) {
                norm_type = 2.0f;
            }
        }
        
        // Create indices tensor for which embeddings to renormalize
        std::vector<int64_t> indices_vec;
        int num_indices = 1;
        
        if (offset + sizeof(uint8_t) <= Size) {
            num_indices = (Data[offset] % 10) + 1;
            offset += sizeof(uint8_t);
        }
        
        // Generate indices within the valid range
        int64_t num_embeddings = weight.size(0);
        
        for (int i = 0; i < num_indices && offset + sizeof(uint8_t) <= Size; i++) {
            int64_t idx = Data[offset] % std::max<int64_t>(1, num_embeddings);
            offset += sizeof(uint8_t);
            indices_vec.push_back(idx);
        }
        
        // If we couldn't extract any indices, create a default one
        if (indices_vec.empty()) {
            indices_vec.push_back(0);
        }
        
        // Create indices tensor
        torch::Tensor indices = torch::tensor(indices_vec, torch::kInt64);
        
        // Apply embedding_renorm_ (in-place operation)
        try {
            torch::embedding_renorm_(weight, indices, max_norm, norm_type);
        } catch (...) {
            // Silently catch expected failures (shape mismatches, etc.)
        }
        
        // Try with different parameters to increase coverage
        if (offset + sizeof(uint8_t) <= Size) {
            float new_max_norm = static_cast<float>(Data[offset] % 100 + 1) / 10.0f;
            offset += sizeof(uint8_t);
            
            // Try with different norm_type values
            float norm_types[] = {1.0f, 2.0f, 0.5f, 3.0f};
            int norm_idx = 0;
            if (offset < Size) {
                norm_idx = Data[offset] % 4;
                offset++;
            }
            
            try {
                torch::embedding_renorm_(weight, indices, new_max_norm, norm_types[norm_idx]);
            } catch (...) {
                // Silently catch expected failures
            }
        }
        
        // Test with contiguous weight tensor
        if (offset < Size && !weight.is_contiguous()) {
            try {
                torch::Tensor contiguous_weight = weight.contiguous();
                torch::embedding_renorm_(contiguous_weight, indices, max_norm, norm_type);
            } catch (...) {
                // Silently catch expected failures
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