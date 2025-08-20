#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the bessel_j0 operation
        torch::Tensor result = torch::special::bessel_j0(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            result.item();
        }
        
        // Try with different input configurations if we have more data
        if (Size - offset >= 2) {
            // Create another tensor with different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply bessel_j0 to the second tensor
            torch::Tensor result2 = torch::special::bessel_j0(input2);
            
            // Ensure computation is performed
            if (result2.defined() && result2.numel() > 0) {
                result2.item();
            }
        }
        
        // Test with edge cases if we have enough data
        if (Size - offset >= 2) {
            // Create a tensor with potentially extreme values
            torch::Tensor edge_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply special scaling to create extreme values
            edge_input = edge_input * 1e10;
            
            // Apply bessel_j0 to the edge case tensor
            torch::Tensor edge_result = torch::special::bessel_j0(edge_input);
            
            // Ensure computation is performed
            if (edge_result.defined() && edge_result.numel() > 0) {
                edge_result.item();
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