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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse diagonal parameter if we have more data
        int64_t diagonal = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&diagonal, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply triu operation
        torch::Tensor result = torch::triu(input, diagonal);
        
        // Try another variant with inplace operation if possible
        if (input.is_floating_point() && input.is_contiguous()) {
            try {
                torch::Tensor input_copy = input.clone();
                input_copy.triu_(diagonal);
            } catch (...) {
                // Ignore exceptions from inplace version
            }
        }
        
        // Try with different diagonal values if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t diagonal2;
            std::memcpy(&diagonal2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            try {
                torch::Tensor result2 = torch::triu(input, diagonal2);
            } catch (...) {
                // Ignore exceptions from different diagonal
            }
        }
        
        // Try with extreme diagonal values
        try {
            torch::Tensor result_large_pos = torch::triu(input, 1000000);
        } catch (...) {
            // Ignore exceptions
        }
        
        try {
            torch::Tensor result_large_neg = torch::triu(input, -1000000);
        } catch (...) {
            // Ignore exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}