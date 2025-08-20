#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 tensors for cosine_similarity
        if (Size < 4) // Minimum bytes needed for basic tensor creation
            return 0;
        
        // Create first input tensor
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor
        if (offset < Size) {
            torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Get dimension parameter for cosine_similarity
            int64_t dim = 1;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            }
            
            // Get eps parameter for cosine_similarity
            double eps = 1e-8;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&eps, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Ensure tensors have compatible shapes for cosine_similarity
            if (x1.dim() > 0 && x2.dim() > 0) {
                // Try to apply cosine_similarity
                try {
                    torch::Tensor result = torch::cosine_similarity(x1, x2, dim, eps);
                } catch (const c10::Error& e) {
                    // Let PyTorch handle the error
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