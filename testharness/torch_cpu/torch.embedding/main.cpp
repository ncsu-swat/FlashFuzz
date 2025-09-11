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
        
        // Need at least some data to proceed
        if (Size < 4) {
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
        
        // Create indices tensor
        torch::Tensor indices;
        if (offset < Size) {
            indices = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices are integers
            if (indices.scalar_type() != torch::kInt && 
                indices.scalar_type() != torch::kLong) {
                indices = indices.to(torch::kLong);
            }
        } else {
            // Create default indices if we don't have enough data
            indices = torch::tensor({0, 1}, torch::kLong);
        }
        
        // Extract embedding parameters from remaining data
        int64_t padding_idx = -1;
        double scale_grad_by_freq = 0.0;
        bool sparse = false;
        
        if (offset + 3 <= Size) {
            // Extract padding_idx (can be negative)
            int64_t raw_padding_idx;
            std::memcpy(&raw_padding_idx, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_idx = raw_padding_idx;
            
            // Extract scale_grad_by_freq as a boolean
            scale_grad_by_freq = (Data[offset++] % 2 == 0) ? 0.0 : 1.0;
            
            // Extract sparse as a boolean
            sparse = (Data[offset++] % 2 == 0) ? false : true;
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
        auto sum = result.sum();
        
        // Try different variants of embedding
        if (offset < Size) {
            // Try with different padding_idx
            padding_idx = static_cast<int64_t>(Data[offset++]) % (weight.size(0) + 1) - 1;
            result = torch::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
            sum += result.sum();
            
            // Try with different scale_grad_by_freq
            scale_grad_by_freq = !scale_grad_by_freq;
            result = torch::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
            sum += result.sum();
            
            // Try with different sparse
            sparse = !sparse;
            result = torch::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
            sum += result.sum();
        }
        
        // Try with empty indices
        if (offset < Size && Data[offset++] % 2 == 0) {
            torch::Tensor empty_indices = torch::empty({0}, torch::kLong);
            result = torch::embedding(weight, empty_indices, padding_idx, scale_grad_by_freq, sparse);
            sum += result.sum();
        }
        
        // Try with indices that might be out of bounds
        if (offset < Size) {
            int64_t num_embeddings = weight.size(0);
            std::vector<int64_t> extreme_indices;
            for (size_t i = 0; i < 4 && offset < Size; i++) {
                int64_t idx = static_cast<int64_t>(Data[offset++]);
                extreme_indices.push_back(idx);
            }
            
            torch::Tensor extreme_indices_tensor = torch::tensor(extreme_indices, torch::kLong);
            
            try {
                result = torch::embedding(weight, extreme_indices_tensor, padding_idx, scale_grad_by_freq, sparse);
                sum += result.sum();
            } catch (const std::exception&) {
                // Expected exception for out-of-bounds indices
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
