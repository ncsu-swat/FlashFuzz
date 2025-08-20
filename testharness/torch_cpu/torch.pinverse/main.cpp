#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply pinverse operation
        torch::Tensor result = torch::pinverse(input);
        
        // Try with rcond parameter if we have more data
        if (offset + sizeof(double) <= Size) {
            double rcond_raw;
            std::memcpy(&rcond_raw, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Apply pinverse with custom rcond
            torch::Tensor result_with_rcond = torch::pinverse(input, rcond_raw);
        }
        
        // Try with negative rcond (edge case)
        if (offset < Size) {
            double negative_rcond = -1.0e-5;
            torch::Tensor result_negative_rcond = torch::pinverse(input, negative_rcond);
        }
        
        // Try with very small rcond (edge case)
        if (offset < Size) {
            double tiny_rcond = 1.0e-30;
            torch::Tensor result_tiny_rcond = torch::pinverse(input, tiny_rcond);
        }
        
        // Try with very large rcond (edge case)
        if (offset < Size) {
            double large_rcond = 1.0e30;
            torch::Tensor result_large_rcond = torch::pinverse(input, large_rcond);
        }
        
        // Try with zero rcond (edge case)
        if (offset < Size) {
            double zero_rcond = 0.0;
            torch::Tensor result_zero_rcond = torch::pinverse(input, zero_rcond);
        }
        
        // Try with NaN rcond (edge case)
        if (offset < Size) {
            double nan_rcond = std::numeric_limits<double>::quiet_NaN();
            torch::Tensor result_nan_rcond = torch::pinverse(input, nan_rcond);
        }
        
        // Try with infinity rcond (edge case)
        if (offset < Size) {
            double inf_rcond = std::numeric_limits<double>::infinity();
            torch::Tensor result_inf_rcond = torch::pinverse(input, inf_rcond);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}