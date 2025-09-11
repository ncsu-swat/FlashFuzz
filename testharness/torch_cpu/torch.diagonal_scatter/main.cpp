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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create src tensor (to be scattered into input)
        torch::Tensor src;
        if (offset < Size) {
            src = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple tensor
            src = torch::ones_like(input);
        }
        
        // Get parameters for diagonal_scatter
        int64_t offset_val = 0;
        int64_t dim1 = 0;
        int64_t dim2 = 1;
        
        // Extract parameters from remaining data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&offset_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim1, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply diagonal_scatter operation
        // diagonal_scatter(input, src, offset, dim1, dim2)
        torch::Tensor result = torch::diagonal_scatter(input, src, offset_val, dim1, dim2);
        
        // Verify the result is a valid tensor
        if (result.defined()) {
            // Access some elements to ensure the operation completed
            if (result.numel() > 0) {
                auto item = result.flatten()[0].item();
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
