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
        
        // Extract tolerance parameter from remaining data if available
        double tol = 1e-5; // Default tolerance
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&tol, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure tolerance is positive and not too small
            tol = std::abs(tol);
            if (tol < 1e-10) tol = 1e-10;
            if (tol > 1.0) tol = 1.0;
        }
        
        // Extract boolean parameter for hermitian flag if available
        bool hermitian = false;
        if (offset < Size) {
            hermitian = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Convert tensor to float if it's an integer type to avoid precision issues
        torch::ScalarType dtype = input.scalar_type();
        if (dtype == torch::kInt8 || dtype == torch::kUInt8 || 
            dtype == torch::kInt16 || dtype == torch::kInt32 || 
            dtype == torch::kInt64 || dtype == torch::kBool) {
            input = input.to(torch::kFloat);
        }
        
        // Call matrix_rank with different parameter combinations
        torch::Tensor result1 = torch::linalg_matrix_rank(input);
        
        // Try with explicit tolerance
        torch::Tensor result2 = torch::linalg_matrix_rank(input, tol);
        
        // Try with hermitian flag
        if (input.dim() >= 2 && input.size(0) == input.size(1)) {
            // Only square matrices can be hermitian
            torch::Tensor result3 = torch::linalg_matrix_rank(input, tol, hermitian);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}