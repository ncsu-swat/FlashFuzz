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
        
        if (Size < 8) {
            return 0;
        }
        
        // Read dimensions from fuzzer data for controlled tensor shapes
        uint8_t m_raw = Data[offset++] % 32 + 1;  // rows: 1-32
        uint8_t n_raw = Data[offset++] % 32 + 1;  // cols of A: 1-32
        uint8_t k_raw = Data[offset++] % 8 + 1;   // cols of B: 1-8
        
        int64_t m = static_cast<int64_t>(m_raw);
        int64_t n = static_cast<int64_t>(n_raw);
        int64_t k = static_cast<int64_t>(k_raw);
        
        // Create coefficient matrix A with shape (m, n)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create right-hand side B
        torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if needed
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat);
        }
        if (!B.is_floating_point()) {
            B = B.to(torch::kFloat);
        }
        
        // Reshape A to be a 2D matrix (m, n)
        int64_t a_numel = A.numel();
        if (a_numel == 0) {
            return 0;
        }
        
        // Adjust dimensions based on available elements
        if (a_numel < m * n) {
            m = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(a_numel)));
            n = std::max<int64_t>(1, a_numel / m);
        }
        A = A.flatten().slice(0, 0, m * n).reshape({m, n});
        
        // Reshape B to be (m, k) - must have same number of rows as A
        int64_t b_numel = B.numel();
        if (b_numel == 0) {
            return 0;
        }
        
        if (b_numel < m * k) {
            k = std::max<int64_t>(1, b_numel / m);
            if (k == 0) k = 1;
        }
        
        int64_t b_elements = std::min(b_numel, m * k);
        if (b_elements < m) {
            // Not enough elements, pad with zeros
            B = torch::zeros({m, k}, torch::kFloat);
            B.flatten().slice(0, 0, b_numel).copy_(B.flatten().slice(0, 0, b_numel));
        } else {
            B = B.flatten().slice(0, 0, m * k).reshape({m, k});
        }
        
        // Ensure same dtype and device
        B = B.to(A.dtype());
        
        // Call torch.linalg.lstsq - this is the C++ equivalent of torch.lstsq (deprecated)
        // linalg_lstsq(A, B) solves min ||A @ X - B||
        try {
            auto result = torch::linalg_lstsq(A, B);
            
            // Access results
            auto solution = std::get<0>(result);     // shape (n, k)
            auto residuals = std::get<1>(result);    // may be empty
            auto rank = std::get<2>(result);         // rank
            auto singular_values = std::get<3>(result);
            
            // Use the solution to prevent optimization
            auto solution_sum = solution.sum();
            
            // residuals may be empty, check before accessing
            if (residuals.numel() > 0) {
                auto residuals_sum = residuals.sum();
                volatile float r = residuals_sum.item<float>();
                (void)r;
            }
            
            volatile float s = solution_sum.item<float>();
            (void)s;
        }
        catch (const c10::Error &e) {
            // Expected errors from invalid matrix configurations - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}