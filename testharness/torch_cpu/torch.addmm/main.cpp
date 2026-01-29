#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For isfinite

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
        // Need enough bytes for dimensions and some tensor data
        if (Size < 12)
            return 0;
        
        size_t offset = 0;
        
        // Extract dimensions for matrix multiplication compatibility
        // mat1: (n, m), mat2: (m, p), input: broadcastable to (n, p)
        uint8_t n_raw = Data[offset++];
        uint8_t m_raw = Data[offset++];
        uint8_t p_raw = Data[offset++];
        
        // Ensure reasonable dimensions (1-32)
        int64_t n = (n_raw % 32) + 1;
        int64_t m = (m_raw % 32) + 1;
        int64_t p = (p_raw % 32) + 1;
        
        // Create tensors with compatible shapes
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor mat1 = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor mat2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape tensors to be compatible for addmm
        // input: (n, p) or broadcastable, mat1: (n, m), mat2: (m, p)
        try {
            input = input.reshape({n, p}).to(torch::kFloat);
            mat1 = mat1.reshape({n, m}).to(torch::kFloat);
            mat2 = mat2.reshape({m, p}).to(torch::kFloat);
        } catch (const std::exception&) {
            // If reshape fails, create new tensors with correct shapes
            input = torch::randn({n, p});
            mat1 = torch::randn({n, m});
            mat2 = torch::randn({m, p});
        }
        
        // Get scalar values for beta and alpha if there's data left
        double beta = 1.0;
        double alpha = 1.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize to avoid NaN/Inf issues
            if (!std::isfinite(beta)) {
                beta = 1.0;
            }
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (!std::isfinite(alpha)) {
                alpha = 1.0;
            }
        }
        
        // Try different variants of addmm
        try {
            // Variant 1: Basic addmm
            auto result1 = torch::addmm(input, mat1, mat2);
        } catch (const std::exception&) {
            // Silent catch for expected failures
        }
        
        try {
            // Variant 2: With beta and alpha
            auto result2 = torch::addmm(input, mat1, mat2, beta, alpha);
        } catch (const std::exception&) {
        }
        
        try {
            // Variant 3: Out variant
            torch::Tensor out = torch::empty({n, p});
            torch::addmm_out(out, input, mat1, mat2);
        } catch (const std::exception&) {
        }
        
        try {
            // Variant 4: Out variant with beta and alpha
            torch::Tensor out = torch::empty({n, p});
            torch::addmm_out(out, input, mat1, mat2, beta, alpha);
        } catch (const std::exception&) {
        }
        
        try {
            // Variant 5: Method variant
            auto result5 = input.addmm(mat1, mat2);
        } catch (const std::exception&) {
        }
        
        try {
            // Variant 6: Method variant with beta and alpha
            auto result6 = input.addmm(mat1, mat2, beta, alpha);
        } catch (const std::exception&) {
        }
        
        // Try in-place variant
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.addmm_(mat1, mat2);
        } catch (const std::exception&) {
        }
        
        try {
            // In-place with beta and alpha
            torch::Tensor input_copy = input.clone();
            input_copy.addmm_(mat1, mat2, beta, alpha);
        } catch (const std::exception&) {
        }
        
        // Test with 1-D input (should broadcast)
        try {
            torch::Tensor input_1d = torch::randn({p});
            auto result = torch::addmm(input_1d, mat1, mat2);
        } catch (const std::exception&) {
        }
        
        // Test with scalar input (should broadcast)
        try {
            torch::Tensor input_scalar = torch::randn({1});
            auto result = torch::addmm(input_scalar, mat1, mat2);
        } catch (const std::exception&) {
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}