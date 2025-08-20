#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a value to use as sections parameter
        int64_t sections = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&sections, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sections a reasonable value
            sections = std::abs(sections) % 10 + 1;
        } else {
            // Default value if not enough data
            sections = 2;
        }
        
        // Try hsplit with sections parameter
        try {
            std::vector<torch::Tensor> result1 = torch::hsplit(input_tensor, sections);
        } catch (...) {
            // Silently catch exceptions from hsplit with sections
        }
        
        // Try hsplit with indices parameter
        if (offset + sizeof(int64_t) <= Size) {
            std::vector<int64_t> indices;
            int64_t num_indices = std::abs(*(int64_t*)(Data + offset)) % 5 + 1;
            offset += sizeof(int64_t);
            
            for (int64_t i = 0; i < num_indices && offset + sizeof(int64_t) <= Size; i++) {
                int64_t idx;
                std::memcpy(&idx, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                indices.push_back(idx);
            }
            
            if (!indices.empty()) {
                try {
                    std::vector<torch::Tensor> result2 = torch::hsplit(input_tensor, indices);
                } catch (...) {
                    // Silently catch exceptions from hsplit with indices
                }
            }
        }
        
        // Try hsplit on tensors with different shapes and ranks
        if (offset + 4 < Size) {
            // Create another tensor with different properties
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            try {
                // Try with sections
                std::vector<torch::Tensor> result3 = torch::hsplit(another_tensor, 2);
            } catch (...) {
                // Silently catch exceptions
            }
        }
        
        // Try hsplit on edge cases: 0-dim tensor, 1-dim tensor
        try {
            // Create a scalar tensor (0-dim)
            torch::Tensor scalar_tensor = torch::tensor(1.0);
            std::vector<torch::Tensor> result4 = torch::hsplit(scalar_tensor, 1);
        } catch (...) {
            // Silently catch exceptions
        }
        
        try {
            // Create a 1-dim tensor
            torch::Tensor vector_tensor = torch::ones({10});
            std::vector<torch::Tensor> result5 = torch::hsplit(vector_tensor, 2);
        } catch (...) {
            // Silently catch exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}