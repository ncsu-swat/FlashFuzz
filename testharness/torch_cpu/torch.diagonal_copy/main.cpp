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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // diagonal_copy requires at least 2 dimensions
        if (input_tensor.dim() < 2) {
            return 0;
        }
        
        int64_t ndim = input_tensor.dim();
        
        // Extract parameters for diagonal_copy from remaining data
        int64_t offset_value = 0;
        int64_t dim1 = 0;
        int64_t dim2 = 1;
        
        // Parse offset value if we have enough data
        if (offset + sizeof(int8_t) <= Size) {
            // Use smaller type to get reasonable offset values
            int8_t small_offset;
            std::memcpy(&small_offset, Data + offset, sizeof(int8_t));
            offset_value = static_cast<int64_t>(small_offset);
            offset += sizeof(int8_t);
        }
        
        // Parse dim1 if we have enough data - constrain to valid range
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t dim_byte;
            std::memcpy(&dim_byte, Data + offset, sizeof(uint8_t));
            dim1 = static_cast<int64_t>(dim_byte % ndim);
            offset += sizeof(uint8_t);
        }
        
        // Parse dim2 if we have enough data - constrain to valid range and different from dim1
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t dim_byte;
            std::memcpy(&dim_byte, Data + offset, sizeof(uint8_t));
            dim2 = static_cast<int64_t>(dim_byte % ndim);
            offset += sizeof(uint8_t);
        }
        
        // Ensure dim1 != dim2
        if (dim1 == dim2) {
            dim2 = (dim1 + 1) % ndim;
        }
        
        torch::Tensor result;
        
        // Test with all parameters
        try {
            result = torch::diagonal_copy(input_tensor, offset_value, dim1, dim2);
            
            // Basic sanity check on result
            if (result.defined() && result.numel() > 0) {
                auto item = result.flatten()[0].item();
                (void)item;
            }
        } catch (const std::exception&) {
            // Expected for invalid parameter combinations
        }
        
        // Try with default parameters (offset=0, dim1=0, dim2=1)
        try {
            torch::Tensor result_default = torch::diagonal_copy(input_tensor);
            if (result_default.defined() && result_default.numel() > 0) {
                auto item = result_default.flatten()[0].item();
                (void)item;
            }
        } catch (const std::exception&) {
            // Expected for some tensor configurations
        }
        
        // Try with negative offset
        try {
            torch::Tensor result_neg = torch::diagonal_copy(input_tensor, -offset_value, dim1, dim2);
            (void)result_neg;
        } catch (const std::exception&) {
            // Expected for invalid offsets
        }
        
        // Try with swapped dimensions
        try {
            torch::Tensor result_swap = torch::diagonal_copy(input_tensor, offset_value, dim2, dim1);
            (void)result_swap;
        } catch (const std::exception&) {
            // Expected for some configurations
        }
        
        // Try with only offset parameter specified
        try {
            torch::Tensor result_offset_only = torch::diagonal_copy(input_tensor, offset_value);
            (void)result_offset_only;
        } catch (const std::exception&) {
            // Expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}