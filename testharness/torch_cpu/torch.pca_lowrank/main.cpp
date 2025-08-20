#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for pca_lowrank if there's more data
        int64_t q = 6; // Default value
        bool center = false;
        bool compute_uv = true;
        
        if (offset + 3 <= Size) {
            // Extract q parameter (number of principal components)
            uint8_t q_byte = Data[offset++];
            q = static_cast<int64_t>(q_byte) % 10 + 1; // Keep q reasonable (1-10)
            
            // Extract boolean parameters
            center = Data[offset++] & 0x1;
            compute_uv = Data[offset++] & 0x1;
        }
        
        // Call pca_lowrank
        try {
            auto result = torch::linalg_pca_lowrank(input, q, center, compute_uv);
            
            // Unpack the result (U, S, V)
            auto U = std::get<0>(result);
            auto S = std::get<1>(result);
            auto V = std::get<2>(result);
            
            // Perform some basic operations on the results to ensure they're used
            if (compute_uv) {
                auto reconstructed = U.matmul(torch::diag(S)).matmul(V.transpose(0, 1));
                auto diff = torch::norm(reconstructed - input);
            } else {
                auto norm_S = torch::norm(S);
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but don't discard the input
            return 0;
        }
        
        // Try with different parameters if there's enough data
        if (offset + 1 <= Size) {
            bool new_center = Data[offset++] & 0x1;
            
            try {
                // Call with different parameters
                auto result2 = torch::linalg_pca_lowrank(input, q, new_center, !compute_uv);
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors but don't discard the input
                return 0;
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