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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor A (matrix)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure A has at least 2 dimensions for QR decomposition
        if (A.dim() < 2) {
            if (A.dim() == 0) {
                A = A.unsqueeze(0).unsqueeze(0);
            } else {
                A = A.unsqueeze(0);
            }
        }
        
        // Create tau tensor (vector of scalar factors)
        torch::Tensor tau;
        if (offset < Size) {
            tau = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure tau is 1D and has appropriate length
            if (tau.dim() != 1) {
                tau = tau.flatten();
            }
            
            // Resize tau to match min dimension of A if needed
            int64_t min_dim = std::min(A.size(0), A.size(1));
            if (tau.size(0) != min_dim) {
                tau = tau.index({torch::indexing::Slice(0, min_dim)});
                if (tau.size(0) < min_dim) {
                    tau = torch::cat({tau, torch::zeros(min_dim - tau.size(0), tau.options())});
                }
            }
        } else {
            // Create a default tau tensor if we don't have enough data
            int64_t min_dim = std::min(A.size(0), A.size(1));
            tau = torch::zeros(min_dim, A.options());
        }
        
        // Create C tensor (input/output matrix)
        torch::Tensor C;
        if (offset < Size) {
            C = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure C has at least 2 dimensions
            if (C.dim() < 2) {
                if (C.dim() == 0) {
                    C = C.unsqueeze(0).unsqueeze(0);
                } else {
                    C = C.unsqueeze(0);
                }
            }
        } else {
            // Create a default C tensor if we don't have enough data
            C = torch::ones({A.size(0), A.size(0)}, A.options());
        }
        
        // Parse left/right flag and transpose flag
        bool left = true;
        bool transpose = false;
        
        if (offset < Size) {
            left = Data[offset++] & 1;
        }
        
        if (offset < Size) {
            transpose = Data[offset++] & 1;
        }
        
        // Try to make tensors have compatible dtypes
        torch::ScalarType common_type = A.scalar_type();
        if (common_type != tau.scalar_type()) {
            if (torch::isFloatingType(common_type) && torch::isFloatingType(tau.scalar_type())) {
                tau = tau.to(common_type);
            } else {
                common_type = torch::kFloat;
                A = A.to(common_type);
                tau = tau.to(common_type);
            }
        }
        
        if (common_type != C.scalar_type()) {
            C = C.to(common_type);
        }
        
        // Ensure we're working with floating point types for QR
        if (!torch::isFloatingType(common_type) && !torch::isComplexType(common_type)) {
            A = A.to(torch::kFloat);
            tau = tau.to(torch::kFloat);
            C = C.to(torch::kFloat);
        }
        
        // Compute QR decomposition to get a valid A and tau
        auto qr_result = torch::linalg_qr(A, "reduced");
        torch::Tensor Q = std::get<0>(qr_result);
        torch::Tensor R = std::get<1>(qr_result);
        
        // Apply ormqr operation
        torch::Tensor result;
        try {
            result = torch::ormqr(A, tau, C, left, transpose);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but let the fuzzer continue
            return 0;
        }
        
        // Verify result is a valid tensor
        if (result.defined() && !result.isnan().any().item<bool>() && 
            !result.isinf().any().item<bool>()) {
            // Optional: perform a simple operation on the result to ensure it's usable
            auto sum = result.sum();
        }
    }
    catch (const std::exception &e)
    {
        return 0; // discard the input
    }
    return 0; // keep the input
}