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
        
        // Create the coefficient tensor A
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the right-hand side tensor B
        if (offset < Size) {
            torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Extract dims parameter from the remaining data
            int64_t dims = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dims, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            }
            
            // Try to make A and B compatible for tensorsolve
            // For tensorsolve, we need prod(A.shape) = prod(B.shape)
            // and B.shape should be the first dims dimensions of A.shape
            
            // Attempt the operation with various configurations
            try {
                // Try with default dims (None in Python)
                auto result = torch::linalg_tensorsolve(A, B);
            } catch (...) {
                // Ignore exceptions from the default case
            }
            
            try {
                // Try with explicit dims
                auto result = torch::linalg_tensorsolve(A, B, dims);
            } catch (...) {
                // Ignore exceptions from the explicit dims case
            }
            
            // Try with different dims values
            if (A.dim() > 0) {
                for (int i = -A.dim(); i <= A.dim(); i++) {
                    try {
                        auto result = torch::linalg_tensorsolve(A, B, i);
                    } catch (...) {
                        // Ignore exceptions
                    }
                }
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