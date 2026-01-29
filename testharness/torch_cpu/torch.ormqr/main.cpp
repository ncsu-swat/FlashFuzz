#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 16) {
            return -1;
        }
        
        // Parse dimensions from fuzzer data for controlled tensor creation
        uint8_t m_raw = Data[offset++] % 16 + 1;  // 1-16
        uint8_t n_raw = Data[offset++] % 16 + 1;  // 1-16
        uint8_t k_raw = Data[offset++] % 16 + 1;  // 1-16
        
        int64_t m = static_cast<int64_t>(m_raw);
        int64_t n = static_cast<int64_t>(n_raw);
        int64_t k = std::min(static_cast<int64_t>(k_raw), std::min(m, n));
        
        // Parse flags
        bool left = (offset < Size) ? (Data[offset++] & 1) : true;
        bool transpose = (offset < Size) ? (Data[offset++] & 1) : false;
        
        // Determine dtype (float or double for real, complex64 or complex128 for complex)
        uint8_t dtype_selector = (offset < Size) ? Data[offset++] % 4 : 0;
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat; break;
            case 1: dtype = torch::kDouble; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
            default: dtype = torch::kFloat; break;
        }
        
        // Create input matrix A for geqrf (m x n)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape and convert A to appropriate shape and type
        A = A.flatten();
        int64_t needed_elements = m * n;
        if (A.numel() < needed_elements) {
            A = torch::cat({A, torch::zeros(needed_elements - A.numel())});
        }
        A = A.index({torch::indexing::Slice(0, needed_elements)}).reshape({m, n}).to(dtype);
        
        // Compute QR factorization using geqrf to get householder reflectors and tau
        torch::Tensor input_qr, tau;
        try {
            auto geqrf_result = torch::geqrf(A);
            input_qr = std::get<0>(geqrf_result);  // Contains reflectors in lower triangle
            tau = std::get<1>(geqrf_result);       // Scalar factors
        } catch (...) {
            return -1;
        }
        
        // Create matrix C with appropriate dimensions
        // For left=true: C is (m x nrhs), for left=false: C is (nrhs x m)
        int64_t nrhs = std::max(int64_t(1), k);
        int64_t c_rows = left ? m : nrhs;
        int64_t c_cols = left ? nrhs : m;
        
        torch::Tensor C = fuzzer_utils::createTensor(Data, Size, offset);
        C = C.flatten();
        int64_t c_needed = c_rows * c_cols;
        if (C.numel() < c_needed) {
            C = torch::cat({C, torch::zeros(c_needed - C.numel())});
        }
        C = C.index({torch::indexing::Slice(0, c_needed)}).reshape({c_rows, c_cols}).to(dtype);
        
        // Call torch::ormqr
        // ormqr(input, tau, other, left, transpose)
        // input: (m, n) matrix containing Householder reflectors
        // tau: (k,) vector of scalar factors where k = min(m, n)
        // other: matrix to multiply, (m, nrhs) if left=true, (nrhs, m) if left=false
        torch::Tensor result;
        try {
            result = torch::ormqr(input_qr, tau, C, left, transpose);
        } catch (...) {
            // Shape mismatches or other expected errors
            return 0;
        }
        
        // Verify result is valid
        if (result.defined()) {
            // Access the result to ensure computation completed
            volatile auto numel = result.numel();
            (void)numel;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}