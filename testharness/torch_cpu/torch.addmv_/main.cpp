#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need enough data for dimensions and values
        if (Size < 4)
            return 0;
        
        size_t offset = 0;
        
        // Extract dimensions from fuzzer data for compatible shapes
        // addmv_: self (n,), mat (n, m), vec (m,)
        uint8_t n_raw = Data[offset++];
        uint8_t m_raw = Data[offset++];
        
        // Ensure reasonable dimensions (1-64)
        int64_t n = (n_raw % 64) + 1;
        int64_t m = (m_raw % 64) + 1;
        
        // Extract dtype indicator
        uint8_t dtype_indicator = 0;
        if (offset < Size) {
            dtype_indicator = Data[offset++];
        }
        
        // Choose dtype based on fuzzer data
        torch::ScalarType dtype;
        switch (dtype_indicator % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create tensors with compatible shapes
        auto options = torch::TensorOptions().dtype(dtype);
        
        // self: 1D tensor of size n
        torch::Tensor self = torch::randn({n}, options);
        
        // mat: 2D matrix of size (n, m)
        torch::Tensor mat = torch::randn({n, m}, options);
        
        // vec: 1D vector of size m
        torch::Tensor vec = torch::randn({m}, options);
        
        // Get beta and alpha values from fuzzer data
        double beta = 1.0;
        double alpha = 1.0;
        
        if (offset + sizeof(float) <= Size) {
            float beta_f;
            std::memcpy(&beta_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp to reasonable range to avoid numerical issues
            if (std::isfinite(beta_f)) {
                beta = static_cast<double>(beta_f);
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float alpha_f;
            std::memcpy(&alpha_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(alpha_f)) {
                alpha = static_cast<double>(alpha_f);
            }
        }
        
        // Create a copy of self for verification
        torch::Tensor self_copy = self.clone();
        
        // Apply the addmv_ operation (in-place)
        // self = beta * self + alpha * (mat @ vec)
        self.addmv_(mat, vec, beta, alpha);
        
        // Verify the operation with non-in-place version
        torch::Tensor expected = self_copy.addmv(mat, vec, beta, alpha);
        
        // Check if the in-place operation produced the expected result
        if (self.defined() && expected.defined()) {
            try {
                // Use appropriate tolerance for float16
                double rtol = (dtype == torch::kFloat16) ? 1e-2 : 1e-5;
                double atol = (dtype == torch::kFloat16) ? 1e-2 : 1e-8;
                bool equal = torch::allclose(self, expected, rtol, atol);
                if (!equal) {
                    std::cerr << "In-place and out-of-place operations produced different results" << std::endl;
                }
            } catch (const std::exception& e) {
                // Comparison might fail for certain dtypes
            }
        }
        
        // Additional test: try with zero beta (ignores original self values)
        if (offset < Size && (Data[offset] % 2 == 0)) {
            torch::Tensor self2 = torch::randn({n}, options);
            try {
                self2.addmv_(mat, vec, 0.0, alpha);
            } catch (const std::exception& e) {
                // Some configurations may fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}