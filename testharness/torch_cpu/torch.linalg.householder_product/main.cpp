#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for householder_product
        // The function expects a matrix v of shape (*, m, n) and tau of shape (*, min(m, n))
        torch::Tensor v = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2 dimensions for v
        if (v.dim() < 2) {
            // Add dimensions if needed
            if (v.dim() == 0) {
                v = v.unsqueeze(0).unsqueeze(0);
            } else if (v.dim() == 1) {
                v = v.unsqueeze(0);
            }
        }
        
        // Create tau tensor with appropriate shape
        torch::Tensor tau;
        if (offset < Size) {
            tau = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure tau has correct shape (*, min(m, n))
            int64_t m = v.size(-2);
            int64_t n = v.size(-1);
            int64_t min_mn = std::min(m, n);
            
            // Reshape tau to have the correct last dimension
            if (tau.dim() > 0) {
                std::vector<int64_t> tau_shape = tau.sizes().vec();
                if (tau_shape.size() > 0) {
                    tau_shape[tau_shape.size() - 1] = min_mn;
                    tau = tau.reshape(tau_shape);
                }
            }
            
            // If tau is a scalar, reshape it to match the batch dimensions of v
            if (tau.dim() == 0) {
                std::vector<int64_t> new_shape;
                for (int64_t i = 0; i < v.dim() - 2; i++) {
                    new_shape.push_back(v.size(i));
                }
                new_shape.push_back(min_mn);
                tau = tau.expand(new_shape);
            }
        } else {
            // If we don't have enough data for tau, create a default one
            int64_t m = v.size(-2);
            int64_t n = v.size(-1);
            int64_t min_mn = std::min(m, n);
            
            std::vector<int64_t> tau_shape;
            for (int64_t i = 0; i < v.dim() - 2; i++) {
                tau_shape.push_back(v.size(i));
            }
            tau_shape.push_back(min_mn);
            
            tau = torch::ones(tau_shape, v.options());
        }
        
        // Try to match dtypes between v and tau
        if (v.dtype() != tau.dtype()) {
            tau = tau.to(v.dtype());
        }
        
        // Apply the householder_product operation
        torch::Tensor result = torch::linalg_householder_product(v, tau);
        
        // Optional: perform some operation on the result to ensure it's used
        auto sum = result.sum().item<double>();
        (void)sum; // Prevent unused variable warning
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}