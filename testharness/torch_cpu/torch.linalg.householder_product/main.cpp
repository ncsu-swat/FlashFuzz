#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions from fuzzer data
        uint8_t m_raw = Data[offset++] % 16 + 1;  // m in [1, 16]
        uint8_t n_raw = Data[offset++] % 16 + 1;  // n in [1, 16]
        
        // Ensure m >= n (required by householder_product)
        int64_t m = std::max(m_raw, n_raw);
        int64_t n = std::min(m_raw, n_raw);
        
        // k must be <= n
        uint8_t k_raw = Data[offset++] % 16 + 1;
        int64_t k = std::min(static_cast<int64_t>(k_raw), n);
        
        // Determine dtype from fuzzer data
        uint8_t dtype_selector = Data[offset++] % 4;
        torch::Dtype dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
            default: dtype = torch::kFloat32; break;
        }
        
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Create v tensor with shape (m, n)
        // For householder_product, v contains the Householder vectors
        torch::Tensor v = torch::randn({m, n}, options);
        
        // Optionally add batch dimensions
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            uint8_t batch_size = (offset < Size) ? (Data[offset++] % 4 + 1) : 2;
            v = v.unsqueeze(0).expand({batch_size, m, n}).clone();
        }
        
        // Create tau tensor with shape (*, k) matching batch dims of v
        std::vector<int64_t> tau_shape;
        for (int64_t i = 0; i < v.dim() - 2; i++) {
            tau_shape.push_back(v.size(i));
        }
        tau_shape.push_back(k);
        
        torch::Tensor tau = torch::randn(tau_shape, options);
        
        // Use remaining fuzzer data to perturb tensors
        if (offset < Size) {
            size_t remaining = Size - offset;
            float scale = static_cast<float>(Data[offset] % 100) / 10.0f + 0.1f;
            v = v * scale;
            if (remaining > 1) {
                float tau_scale = static_cast<float>(Data[offset + 1] % 100) / 10.0f + 0.1f;
                tau = tau * tau_scale;
            }
        }
        
        // Apply the householder_product operation
        // This computes the first n columns of a product of Householder matrices
        torch::Tensor result;
        try {
            result = torch::linalg_householder_product(v, tau);
        } catch (const c10::Error&) {
            // Shape mismatches or invalid inputs are expected during fuzzing
            return 0;
        }
        
        // Verify result properties
        if (result.defined()) {
            // Result should have shape (*, m, n)
            auto sum = result.sum();
            (void)sum;
            
            // Check result dimensions match input
            if (result.size(-2) != m || result.size(-1) != n) {
                std::cerr << "Unexpected result shape" << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}