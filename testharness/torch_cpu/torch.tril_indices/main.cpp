#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        size_t offset = 0;
        
        // Need at least enough bytes for two int16_t (row, col), one int8_t (tril_offset), and one byte (dtype)
        if (Size < 6) {
            return 0;
        }
        
        // Parse row and column dimensions using smaller types to avoid extreme values
        // Use int16_t and constrain to reasonable range to avoid memory issues
        int16_t row_raw = 0;
        int16_t col_raw = 0;
        
        std::memcpy(&row_raw, Data + offset, sizeof(int16_t));
        offset += sizeof(int16_t);
        
        std::memcpy(&col_raw, Data + offset, sizeof(int16_t));
        offset += sizeof(int16_t);
        
        // Constrain row and col to reasonable range [0, 1000] to avoid OOM
        int64_t row = static_cast<int64_t>(std::abs(row_raw)) % 1001;
        int64_t col = static_cast<int64_t>(std::abs(col_raw)) % 1001;
        
        // Parse offset parameter - can be negative
        int8_t tril_offset_raw = 0;
        std::memcpy(&tril_offset_raw, Data + offset, sizeof(int8_t));
        offset += sizeof(int8_t);
        int64_t tril_offset = static_cast<int64_t>(tril_offset_raw);
        
        // Parse dtype - tril_indices only supports integer types (kInt32, kInt64)
        torch::ScalarType dtype = (Data[offset++] % 2 == 0) ? torch::kInt32 : torch::kInt64;
        
        // Call tril_indices with basic parameters
        try {
            auto result = torch::tril_indices(row, col, tril_offset);
            
            // Access elements of the result to ensure it's valid
            if (result.numel() > 0) {
                auto first_element = result[0][0];
                (void)first_element;
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with explicit device
        try {
            auto result = torch::tril_indices(row, col, tril_offset, 
                                             torch::TensorOptions().device(torch::kCPU));
            (void)result;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with explicit dtype
        try {
            auto result = torch::tril_indices(row, col, tril_offset, 
                                             torch::TensorOptions().dtype(dtype));
            (void)result;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with both device and dtype
        try {
            auto result = torch::tril_indices(row, col, tril_offset, 
                                             torch::TensorOptions().device(torch::kCPU).dtype(dtype));
            (void)result;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with zero row
        try {
            auto result = torch::tril_indices(0, col, tril_offset);
            (void)result;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with zero col
        try {
            auto result = torch::tril_indices(row, 0, tril_offset);
            (void)result;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with zero dimensions
        try {
            auto result = torch::tril_indices(0, 0, tril_offset);
            (void)result;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with large positive offset (larger than matrix dimensions)
        try {
            auto result = torch::tril_indices(row, col, row + col);
            (void)result;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with large negative offset
        try {
            auto result = torch::tril_indices(row, col, -(row + col));
            (void)result;
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}