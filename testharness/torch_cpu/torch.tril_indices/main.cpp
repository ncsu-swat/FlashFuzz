#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 4 bytes for row, col, offset, and dtype
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
        
        // Parse offset parameter
        int64_t tril_offset = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&tril_offset, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dtype
        torch::ScalarType dtype = torch::kInt64;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Call tril_indices with various combinations
        try {
            auto result = torch::tril_indices(row, col, tril_offset);
            
            // Try to access elements of the result to ensure it's valid
            if (result.numel() > 0) {
                auto first_element = result[0][0];
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with different device
        try {
            auto result = torch::tril_indices(row, col, tril_offset, 
                                             torch::TensorOptions().device(torch::kCPU));
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with different dtype
        try {
            auto result = torch::tril_indices(row, col, tril_offset, 
                                             torch::TensorOptions().dtype(dtype));
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with both device and dtype
        try {
            auto result = torch::tril_indices(row, col, tril_offset, 
                                             torch::TensorOptions().device(torch::kCPU).dtype(dtype));
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with extreme values
        try {
            auto result = torch::tril_indices(std::numeric_limits<int64_t>::max() / 2, 
                                             std::numeric_limits<int64_t>::max() / 2, 
                                             std::numeric_limits<int64_t>::max() / 2);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with negative values
        try {
            auto result = torch::tril_indices(-row, -col, -tril_offset);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with mixed positive/negative values
        try {
            auto result = torch::tril_indices(row, -col, tril_offset);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // Try with zero values
        try {
            auto result = torch::tril_indices(0, 0, tril_offset);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}