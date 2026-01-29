#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least some data to create tensors
        if (Size < 8) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor - for dist to work, tensors should have same number of elements
        // We'll create input2 and then try to reshape it to match input1
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data, use a clone with different values
            input2 = torch::randn_like(input1);
        }
        
        // Get p-norm value from the remaining data
        double p = 2.0; // Default to L2 norm
        if (offset + sizeof(uint8_t) <= Size) {
            // Use a byte to derive p value to avoid NaN/Inf issues
            uint8_t p_byte = Data[offset];
            offset += sizeof(uint8_t);
            // Map byte to reasonable p range: 0.0 to 10.0
            p = (static_cast<double>(p_byte) / 255.0) * 10.0;
        }
        
        // Try operations that may fail due to shape mismatch
        try {
            // Call torch.dist with the parsed p value
            torch::Tensor result = torch::dist(input1, input2, p);
            
            // Try with common p-norm values
            result = torch::dist(input1, input2, 1.0);  // L1 norm
            result = torch::dist(input1, input2, 2.0);  // L2 norm
        } catch (const std::exception&) {
            // Shape mismatch or other expected failures - try with same tensor
        }
        
        // Always test with matching shapes using same tensor
        torch::Tensor result = torch::dist(input1, input1, p);
        
        // Test with L0 norm (count of non-zero differences)
        result = torch::dist(input1, input1, 0.0);
        
        // Test with L-infinity norm
        result = torch::dist(input1, input1, INFINITY);
        
        // Test with fractional p
        result = torch::dist(input1, input1, 0.5);
        result = torch::dist(input1, input1, 1.5);
        
        // Test with a slightly perturbed tensor for more interesting results
        torch::Tensor input1_perturbed = input1 + torch::randn_like(input1) * 0.1;
        result = torch::dist(input1, input1_perturbed, 2.0);
        result = torch::dist(input1, input1_perturbed, p);
        
        // Test negative p in inner try-catch (expected to fail or have special behavior)
        try {
            result = torch::dist(input1, input1, -1.0);
        } catch (const std::exception&) {
            // Expected exception for negative p
        }
        
        // Test with very small positive p
        try {
            result = torch::dist(input1, input1, 0.001);
        } catch (const std::exception&) {
            // May have numerical issues
        }
        
        // Test with large p approaching infinity
        try {
            result = torch::dist(input1, input1, 100.0);
        } catch (const std::exception&) {
            // May have numerical issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}