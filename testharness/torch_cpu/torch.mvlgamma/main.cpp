#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Create input tensor - mvlgamma works on floating point tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a floating point tensor
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract p parameter from the remaining data
        int64_t p = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure p is within a reasonable range (1 to 10)
            // mvlgamma requires p >= 1
            p = (std::abs(p) % 10) + 1;
        }
        
        // Apply mvlgamma operation
        torch::Tensor result;
        
        // Try different variants of the operation
        if (offset < Size) {
            uint8_t variant = Data[offset++];
            
            if (variant % 2 == 0) {
                // Variant 1: Use the functional API
                result = torch::mvlgamma(input, p);
            } else {
                // Variant 2: Use the tensor method (in-place returns a new tensor)
                result = input.mvlgamma(p);
            }
        } else {
            // Default to functional API if no more data
            result = torch::mvlgamma(input, p);
        }
        
        // Perform some operation with the result to ensure it's used
        // Use try-catch for potential NaN/Inf issues during sum
        try {
            auto sum = result.sum();
            
            // Prevent compiler from optimizing away the computation
            if (sum.item<double>() == -12345.6789) {
                return 1;
            }
        } catch (...) {
            // Silently handle NaN/Inf issues from invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}