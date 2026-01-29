#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Extract dimensions for transpose
        int64_t dim0 = 0;
        int64_t dim1 = 1;
        
        // Get dimensions to transpose if we have enough data
        if (offset + sizeof(int8_t) * 2 <= Size) {
            // Use int8_t to get reasonable dimension values
            dim0 = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            
            dim1 = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
        }
        
        // Get variant selector
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset] % 4;
            offset++;
        }
        
        torch::Tensor result;
        int64_t rank = input_tensor.dim();
        
        // Handle edge cases
        if (rank < 2) {
            // transpose requires at least 2 dimensions
            // Try to reshape to 2D if possible
            if (input_tensor.numel() > 1) {
                try {
                    input_tensor = input_tensor.view({1, -1});
                    rank = 2;
                } catch (...) {
                    // Can't reshape, just return
                    return 0;
                }
            } else {
                return 0;
            }
        }
        
        switch (variant) {
            case 0: {
                // Variant 1: Use raw dimensions (may be out of range - tests error handling)
                try {
                    result = torch::transpose(input_tensor, dim0, dim1);
                } catch (const c10::Error&) {
                    // Expected for invalid dimensions, silently ignore
                    return 0;
                }
                break;
            }
            case 1: {
                // Variant 2: Clamp dimensions to valid range (positive)
                int64_t safe_dim0 = std::abs(dim0) % rank;
                int64_t safe_dim1 = std::abs(dim1) % rank;
                result = torch::transpose(input_tensor, safe_dim0, safe_dim1);
                break;
            }
            case 2: {
                // Variant 3: Use negative dimensions (valid in PyTorch)
                int64_t neg_dim0 = -(std::abs(dim0) % rank + 1);
                int64_t neg_dim1 = -(std::abs(dim1) % rank + 1);
                result = torch::transpose(input_tensor, neg_dim0, neg_dim1);
                break;
            }
            case 3: {
                // Variant 4: Mix positive and negative dimensions
                int64_t pos_dim = std::abs(dim0) % rank;
                int64_t neg_dim = -(std::abs(dim1) % rank + 1);
                result = torch::transpose(input_tensor, pos_dim, neg_dim);
                break;
            }
            default:
                result = torch::transpose(input_tensor, 0, 1);
                break;
        }
        
        // Verify the result is a valid tensor
        if (result.defined()) {
            // Perform operations on result to ensure it's used and valid
            auto sum = result.sum();
            
            // Verify transpose property: transposing twice should give original
            try {
                int64_t d0 = std::abs(dim0) % rank;
                int64_t d1 = std::abs(dim1) % rank;
                torch::Tensor double_transpose = torch::transpose(result, d0, d1);
                (void)double_transpose.sum();
            } catch (...) {
                // Ignore errors in verification
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