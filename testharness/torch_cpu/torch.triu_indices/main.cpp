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
        
        // Need at least 4 bytes for row, col, offset, and diagonal
        if (Size < 4) {
            return 0;
        }
        
        // Parse row and column dimensions
        int64_t row = 0;
        int64_t col = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&row, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&col, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse offset and diagonal parameters
        int64_t offset_param = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&offset_param, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        bool diagonal = false;
        if (offset < Size) {
            diagonal = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Call triu_indices with various combinations
        try {
            auto indices = torch::triu_indices(row, col, offset_param);
        } catch (const c10::Error &e) {
            // Expected exceptions from PyTorch API are fine
        }
        
        // Try with different row/col combinations
        try {
            // Try with negative dimensions
            auto indices_neg_row = torch::triu_indices(-row, col, offset_param);
        } catch (const c10::Error &e) {
            // Expected exceptions from PyTorch API are fine
        }
        
        try {
            // Try with negative dimensions
            auto indices_neg_col = torch::triu_indices(row, -col, offset_param);
        } catch (const c10::Error &e) {
            // Expected exceptions from PyTorch API are fine
        }
        
        try {
            // Try with zero dimensions
            auto indices_zero = torch::triu_indices(0, 0, offset_param);
        } catch (const c10::Error &e) {
            // Expected exceptions from PyTorch API are fine
        }
        
        try {
            // Try with large offset
            auto indices_large_offset = torch::triu_indices(row, col, row + col);
        } catch (const c10::Error &e) {
            // Expected exceptions from PyTorch API are fine
        }
        
        try {
            // Try with negative offset
            auto indices_neg_offset = torch::triu_indices(row, col, -offset_param);
        } catch (const c10::Error &e) {
            // Expected exceptions from PyTorch API are fine
        }
        
        // Try with very large dimensions if there's enough data
        if (Size > offset + 2 * sizeof(int64_t)) {
            int64_t large_row = 0;
            int64_t large_col = 0;
            
            std::memcpy(&large_row, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&large_col, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make them large but not too large to avoid OOM
            large_row = std::abs(large_row) % 1000;
            large_col = std::abs(large_col) % 1000;
            
            try {
                auto indices_large = torch::triu_indices(large_row, large_col, offset_param);
            } catch (const c10::Error &e) {
                // Expected exceptions from PyTorch API are fine
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
