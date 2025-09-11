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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte for atol parameter if available
        double atol = 1e-5;
        if (offset < Size) {
            uint8_t atol_byte = Data[offset++];
            // Scale atol between 1e-10 and 1e-1
            atol = std::pow(10.0, -10.0 + (atol_byte % 10));
        }
        
        // Get a byte for rtol parameter if available
        double rtol = 1e-3;
        if (offset < Size) {
            uint8_t rtol_byte = Data[offset++];
            // Scale rtol between 1e-8 and 1e-1
            rtol = std::pow(10.0, -8.0 + (rtol_byte % 8));
        }
        
        // Get a byte for hermitian parameter if available
        bool hermitian = false;
        if (offset < Size) {
            hermitian = (Data[offset++] % 2) == 1;
        }
        
        // Call matrix_rank with different parameter combinations
        torch::Tensor result;
        
        // Basic call with default parameters
        result = torch::matrix_rank(input);
        
        // Call with tolerance parameters
        if (offset < Size) {
            result = torch::matrix_rank(input, atol);
        }
        
        if (offset < Size) {
            result = torch::matrix_rank(input, atol, rtol);
        }
        
        // Call with hermitian parameter
        if (offset < Size) {
            result = torch::matrix_rank(input, atol, rtol, hermitian);
        }
        
        // Try with different tensor types if we have enough data
        if (offset + 4 < Size) {
            // Create another tensor with different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Try matrix_rank on this tensor too
            torch::Tensor result2 = torch::matrix_rank(input2);
            
            // Try with tolerance parameters
            result2 = torch::matrix_rank(input2, atol, rtol, hermitian);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
