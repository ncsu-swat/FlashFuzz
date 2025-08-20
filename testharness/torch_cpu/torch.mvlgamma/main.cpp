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
        
        // Extract p parameter from the remaining data
        int64_t p = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure p is within a reasonable range
            p = (std::abs(p) % 10) + 1;
        }
        
        // Apply mvlgamma operation
        torch::Tensor result;
        
        // Try different variants of the operation
        if (offset < Size) {
            uint8_t variant = Data[offset++];
            
            if (variant % 3 == 0) {
                // Variant 1: Use the functional API
                result = torch::mvlgamma(input, p);
            } else if (variant % 3 == 1) {
                // Variant 2: Use the in-place API
                result = input.clone();
                result.mvlgamma_(p);
            } else {
                // Variant 3: Use the out variant
                result = torch::empty_like(input);
                torch::mvlgamma_out(result, input, p);
            }
        } else {
            // Default to functional API if no more data
            result = torch::mvlgamma(input, p);
        }
        
        // Optional: Perform some operation with the result to ensure it's used
        auto sum = result.sum();
        
        // Prevent compiler from optimizing away the computation
        if (sum.item<double>() == -12345.6789) {
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}