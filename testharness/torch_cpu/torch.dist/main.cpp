#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create two input tensors for torch.dist
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            // Try to use the same tensor if we don't have enough data
            torch::Tensor result = torch::dist(input1, input1);
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get p-norm value from the remaining data
        double p = 2.0; // Default to L2 norm
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Call torch.dist with different p values
        torch::Tensor result;
        
        // Try with the parsed p value
        result = torch::dist(input1, input2, p);
        
        // Try with common p-norm values
        result = torch::dist(input1, input2, 0.0);  // L0 norm
        result = torch::dist(input1, input2, 1.0);  // L1 norm
        result = torch::dist(input1, input2, 2.0);  // L2 norm
        result = torch::dist(input1, input2, INFINITY);  // L-infinity norm
        
        // Try with negative p (should throw an exception)
        try {
            result = torch::dist(input1, input2, -1.0);
        } catch (const std::exception&) {
            // Expected exception for negative p
        }
        
        // Try with fractional p
        result = torch::dist(input1, input2, 0.5);
        
        // Try with very large p
        if (offset + sizeof(double) <= Size) {
            double large_p;
            std::memcpy(&large_p, Data + offset, sizeof(double));
            try {
                result = torch::dist(input1, input2, large_p);
            } catch (const std::exception&) {
                // May throw for extreme values
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