#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <cstring>
#include <cstdlib>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t data_offset = 0;
        
        // Need at least 3 bytes for row, col, and offset parameters
        if (Size < 3) {
            return 0;
        }
        
        // Parse row and column dimensions from fuzzer data
        // Constrain to reasonable sizes to avoid OOM (0-500)
        int64_t row = 0;
        int64_t col = 0;
        int64_t offset_param = 0;
        
        if (data_offset + sizeof(int16_t) <= Size) {
            int16_t raw_row;
            std::memcpy(&raw_row, Data + data_offset, sizeof(int16_t));
            data_offset += sizeof(int16_t);
            row = std::abs(raw_row) % 501;  // 0-500
        }
        
        if (data_offset + sizeof(int16_t) <= Size) {
            int16_t raw_col;
            std::memcpy(&raw_col, Data + data_offset, sizeof(int16_t));
            data_offset += sizeof(int16_t);
            col = std::abs(raw_col) % 501;  // 0-500
        }
        
        if (data_offset + sizeof(int16_t) <= Size) {
            int16_t raw_offset;
            std::memcpy(&raw_offset, Data + data_offset, sizeof(int16_t));
            data_offset += sizeof(int16_t);
            // Offset can be negative or positive, constrain to reasonable range
            offset_param = raw_offset % 1001 - 500;  // -500 to 500
        }
        
        // Call triu_indices with parsed parameters
        try {
            auto indices = torch::triu_indices(row, col, offset_param);
            // Verify the output has expected shape (2, num_indices)
            if (indices.dim() == 2 && indices.size(0) == 2) {
                // Access elements to ensure tensor is properly formed
                auto row_indices = indices[0];
                auto col_indices = indices[1];
            }
        } catch (const c10::Error &e) {
            // Expected exceptions from PyTorch API are fine
        }
        
        // Try with zero dimensions
        try {
            auto indices_zero = torch::triu_indices(0, 0, 0);
        } catch (const c10::Error &e) {
            // Expected
        }
        
        // Try with zero row
        try {
            auto indices_zero_row = torch::triu_indices(0, col, offset_param);
        } catch (const c10::Error &e) {
            // Expected
        }
        
        // Try with zero col
        try {
            auto indices_zero_col = torch::triu_indices(row, 0, offset_param);
        } catch (const c10::Error &e) {
            // Expected
        }
        
        // Try with large positive offset (beyond matrix)
        try {
            auto indices_large_offset = torch::triu_indices(row, col, static_cast<int64_t>(row + col));
        } catch (const c10::Error &e) {
            // Expected
        }
        
        // Try with large negative offset
        try {
            auto indices_neg_offset = torch::triu_indices(row, col, -static_cast<int64_t>(row + col));
        } catch (const c10::Error &e) {
            // Expected
        }
        
        // Try with different dtype options if we have more data
        if (data_offset < Size) {
            uint8_t dtype_selector = Data[data_offset++] % 2;
            
            try {
                torch::TensorOptions options;
                if (dtype_selector == 0) {
                    options = torch::TensorOptions().dtype(torch::kInt64);
                } else {
                    options = torch::TensorOptions().dtype(torch::kInt32);
                }
                auto indices_typed = torch::triu_indices(row, col, offset_param, options);
            } catch (const c10::Error &e) {
                // Expected
            }
        }
        
        // Try square matrix case
        try {
            int64_t size = (row + col) / 2;
            if (size > 0 && size <= 500) {
                auto indices_square = torch::triu_indices(size, size, offset_param);
            }
        } catch (const c10::Error &e) {
            // Expected
        }
        
        // Try rectangular matrices (wide and tall)
        try {
            if (row > 0 && col > 0) {
                // Wide matrix
                auto indices_wide = torch::triu_indices(row, col * 2 > 500 ? 500 : col * 2, offset_param);
            }
        } catch (const c10::Error &e) {
            // Expected
        }
        
        try {
            if (row > 0 && col > 0) {
                // Tall matrix
                auto indices_tall = torch::triu_indices(row * 2 > 500 ? 500 : row * 2, col, offset_param);
            }
        } catch (const c10::Error &e) {
            // Expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}