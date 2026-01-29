#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a value to use as sections parameter
        int64_t sections = 2;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&sections, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sections a reasonable positive value (1-10)
            sections = std::abs(sections) % 10 + 1;
        }
        
        // hsplit requires tensor to have at least 1 dimension
        // For 1-D tensors, it splits along dim 0
        // For N-D tensors (N >= 2), it splits along dim 1
        
        // Try hsplit with sections parameter
        try {
            std::vector<torch::Tensor> result1 = torch::hsplit(input_tensor, sections);
        } catch (const c10::Error&) {
            // Expected: shape mismatch, invalid sections, etc.
        } catch (const std::runtime_error&) {
            // Expected: runtime errors from invalid parameters
        }
        
        // Try hsplit with indices parameter (IntArrayRef overload)
        if (offset + sizeof(int64_t) <= Size) {
            int64_t num_indices_raw;
            std::memcpy(&num_indices_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            int64_t num_indices = std::abs(num_indices_raw) % 5 + 1;
            std::vector<int64_t> indices;
            
            for (int64_t i = 0; i < num_indices && offset + sizeof(int64_t) <= Size; i++) {
                int64_t idx;
                std::memcpy(&idx, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                // Keep indices reasonable
                idx = std::abs(idx) % 100;
                indices.push_back(idx);
            }
            
            if (!indices.empty()) {
                try {
                    std::vector<torch::Tensor> result2 = torch::hsplit(input_tensor, indices);
                } catch (const c10::Error&) {
                    // Expected: invalid indices
                } catch (const std::runtime_error&) {
                    // Expected: runtime errors
                }
            }
        }
        
        // Try hsplit on tensors with different shapes
        if (offset + 4 < Size) {
            size_t local_offset = 0;
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, local_offset);
            
            try {
                std::vector<torch::Tensor> result3 = torch::hsplit(another_tensor, 2);
            } catch (const c10::Error&) {
                // Expected
            } catch (const std::runtime_error&) {
                // Expected
            }
        }
        
        // Test with specific tensor shapes that are valid for hsplit
        // 1-D tensor: splits along dim 0
        try {
            int64_t vec_size = (Size > 0 ? (Data[0] % 10 + 2) : 4);
            torch::Tensor vector_tensor = torch::ones({vec_size});
            int64_t split_sections = (Size > 1 ? (Data[1] % 3 + 1) : 2);
            // Ensure sections divides evenly
            if (vec_size % split_sections == 0) {
                std::vector<torch::Tensor> result4 = torch::hsplit(vector_tensor, split_sections);
            }
        } catch (const c10::Error&) {
            // Expected
        } catch (const std::runtime_error&) {
            // Expected
        }
        
        // 2-D tensor: splits along dim 1
        try {
            int64_t rows = (Size > 2 ? (Data[2] % 5 + 1) : 3);
            int64_t cols = (Size > 3 ? (Data[3] % 8 + 2) : 6);
            torch::Tensor matrix_tensor = torch::ones({rows, cols});
            int64_t split_sections = (Size > 4 ? (Data[4] % 3 + 1) : 2);
            // Ensure sections divides evenly along dim 1
            if (cols % split_sections == 0) {
                std::vector<torch::Tensor> result5 = torch::hsplit(matrix_tensor, split_sections);
            }
        } catch (const c10::Error&) {
            // Expected
        } catch (const std::runtime_error&) {
            // Expected
        }
        
        // 3-D tensor: splits along dim 1
        try {
            int64_t d0 = (Size > 5 ? (Data[5] % 3 + 1) : 2);
            int64_t d1 = (Size > 6 ? (Data[6] % 6 + 2) : 4);
            int64_t d2 = (Size > 7 ? (Data[7] % 4 + 1) : 3);
            torch::Tensor tensor_3d = torch::ones({d0, d1, d2});
            int64_t split_sections = (Size > 8 ? (Data[8] % 2 + 1) : 2);
            if (d1 % split_sections == 0) {
                std::vector<torch::Tensor> result6 = torch::hsplit(tensor_3d, split_sections);
            }
        } catch (const c10::Error&) {
            // Expected
        } catch (const std::runtime_error&) {
            // Expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}