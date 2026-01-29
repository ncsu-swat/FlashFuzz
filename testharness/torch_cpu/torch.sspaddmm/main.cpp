#include "fuzzer_utils.h"
#include <iostream>

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
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions from fuzzer data
        uint8_t n_raw = Data[offset++] % 16 + 1;  // 1-16
        uint8_t k_raw = Data[offset++] % 16 + 1;  // 1-16
        uint8_t m_raw = Data[offset++] % 16 + 1;  // 1-16
        
        int64_t n = static_cast<int64_t>(n_raw);
        int64_t k = static_cast<int64_t>(k_raw);
        int64_t m = static_cast<int64_t>(m_raw);

        // Extract scalar values for beta and alpha from fuzzer data
        float beta = 1.0f;
        float alpha = 1.0f;
        
        if (offset + 4 <= Size) {
            int32_t beta_int;
            memcpy(&beta_int, Data + offset, sizeof(int32_t));
            offset += 4;
            beta = static_cast<float>(beta_int % 100) / 10.0f;
        }
        
        if (offset + 4 <= Size) {
            int32_t alpha_int;
            memcpy(&alpha_int, Data + offset, sizeof(int32_t));
            offset += 4;
            alpha = static_cast<float>(alpha_int % 100) / 10.0f;
        }

        // Create mat1 (n x k) and mat2 (k x m) as dense tensors
        torch::Tensor mat1 = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor mat2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape to required dimensions for matrix multiplication
        try {
            mat1 = torch::randn({n, k}, torch::kFloat32);
            mat2 = torch::randn({k, m}, torch::kFloat32);
            
            // Use fuzzer data to influence tensor values if available
            if (offset < Size) {
                float scale = static_cast<float>(Data[offset++] % 10 + 1) / 5.0f;
                mat1 = mat1 * scale;
            }
            if (offset < Size) {
                float scale = static_cast<float>(Data[offset++] % 10 + 1) / 5.0f;
                mat2 = mat2 * scale;
            }
        } catch (...) {
            // Silently handle tensor creation failures
            return 0;
        }

        // Create sparse tensor of shape (n x m) - result of mat1 @ mat2 must match
        torch::Tensor sparse;
        try {
            // Create a sparse tensor by making a dense tensor and converting
            torch::Tensor dense_sparse = torch::randn({n, m}, torch::kFloat32);
            
            // Make it sparse by zeroing out some elements based on fuzzer data
            if (offset < Size) {
                float sparsity = static_cast<float>(Data[offset++] % 90 + 10) / 100.0f;
                torch::Tensor mask = torch::rand({n, m}) > sparsity;
                dense_sparse = dense_sparse * mask.to(torch::kFloat32);
            }
            
            sparse = dense_sparse.to_sparse();
        } catch (...) {
            // Silently handle sparse tensor creation failures
            return 0;
        }

        // Apply sspaddmm operation: beta * sparse + alpha * (mat1 @ mat2)
        try {
            torch::Tensor result = torch::sspaddmm(sparse, mat1, mat2, beta, alpha);
            
            // Verify result is sparse
            if (result.is_sparse()) {
                // Access some properties to ensure computation completed
                auto nnz = result._nnz();
                (void)nnz;
            }
        } catch (...) {
            // Silently catch expected failures (e.g., dimension mismatches)
        }

        // Try with default beta and alpha values
        try {
            torch::Tensor result2 = torch::sspaddmm(sparse, mat1, mat2);
            (void)result2;
        } catch (...) {
            // Silently catch expected failures
        }

        // Try with only alpha specified (beta defaults to 1)
        try {
            torch::Tensor result3 = torch::sspaddmm(sparse, mat1, mat2, 1.0, alpha);
            (void)result3;
        } catch (...) {
            // Silently catch expected failures
        }

        // Test with different sparse densities
        try {
            // Create a very sparse tensor
            torch::Tensor indices = torch::randint(0, std::min(n, m), {2, 2}, torch::kLong);
            indices[0] = indices[0] % n;
            indices[1] = indices[1] % m;
            torch::Tensor values = torch::randn({2}, torch::kFloat32);
            torch::Tensor very_sparse = torch::sparse_coo_tensor(indices, values, {n, m});
            
            torch::Tensor result4 = torch::sspaddmm(very_sparse, mat1, mat2, beta, alpha);
            (void)result4;
        } catch (...) {
            // Silently catch expected failures
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}