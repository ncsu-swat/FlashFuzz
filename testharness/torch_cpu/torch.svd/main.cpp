#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for SVD
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // SVD requires at least 2D tensor
        if (input_tensor.dim() < 2) {
            // Add dimensions if needed
            if (input_tensor.dim() == 0) {
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0);
            } else if (input_tensor.dim() == 1) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Get some parameters for SVD from the remaining data
        bool some_bool = false;
        bool compute_uv = true;
        
        if (offset < Size) {
            some_bool = Data[offset++] & 0x1;
            if (offset < Size) {
                compute_uv = Data[offset++] & 0x1;
            }
        }
        
        // Apply SVD operation using torch::svd
        auto svd_result = torch::svd(input_tensor, some_bool, compute_uv);
        
        // Unpack the results
        auto& U = std::get<0>(svd_result);
        auto& S = std::get<1>(svd_result);
        auto& V = std::get<2>(svd_result);
        
        // Perform some operations with the results to ensure they're used
        if (compute_uv) {
            // Reconstruct the original matrix to verify SVD
            auto S_diag = torch::diag_embed(S);
            
            // Adjust dimensions for matrix multiplication if needed
            if (U.dim() > 1 && S_diag.dim() > 1 && V.dim() > 1) {
                // Transpose V to get V^T for reconstruction
                auto V_t = V.transpose(-2, -1);
                
                // Reconstruct: U * S * V^T
                auto reconstructed = torch::matmul(torch::matmul(U, S_diag), V_t);
                
                // Check if the reconstruction is close to the original
                auto diff = torch::abs(reconstructed - input_tensor).max().item<double>();
            }
        }
        
        // Try another variant of SVD
        if (offset < Size && (Data[offset++] & 0x1)) {
            // Use the alternative API
            auto [U2, S2, Vh2] = torch::svd(input_tensor, some_bool);
            
            // Do something with the results
            auto sum_s = S2.sum().item<double>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}