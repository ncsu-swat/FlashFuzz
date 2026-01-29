#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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

        // Extract dimensions for controlled tensor creation
        uint8_t m_val = Data[offset++] % 16 + 1;  // 1-16 rows
        uint8_t n_val = Data[offset++] % 16 + 1;  // 1-16 cols for A
        uint8_t k_val = Data[offset++] % 8 + 1;   // 1-8 cols for B
        uint8_t driver_selector = Data[offset++];
        uint8_t rcond_byte = Data[offset++];
        
        int64_t m = static_cast<int64_t>(m_val);
        int64_t n = static_cast<int64_t>(n_val);
        int64_t k = static_cast<int64_t>(k_val);

        // Create matrix A of shape (m, n)
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape or create A with proper dimensions
        try {
            int64_t a_numel = m * n;
            if (A.numel() < a_numel) {
                A = torch::randn({m, n}, torch::kFloat64);
            } else {
                A = A.flatten().slice(0, 0, a_numel).reshape({m, n}).to(torch::kFloat64);
            }
        } catch (...) {
            A = torch::randn({m, n}, torch::kFloat64);
        }

        // Create matrix B of shape (m, k)
        torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
        
        try {
            int64_t b_numel = m * k;
            if (B.numel() < b_numel) {
                B = torch::randn({m, k}, torch::kFloat64);
            } else {
                B = B.flatten().slice(0, 0, b_numel).reshape({m, k}).to(torch::kFloat64);
            }
        } catch (...) {
            B = torch::randn({m, k}, torch::kFloat64);
        }

        // Select driver - use c10::string_view for the driver parameter
        c10::optional<c10::string_view> driver = c10::nullopt;
        switch (driver_selector % 5) {
            case 0:
                driver = c10::nullopt;
                break;
            case 1:
                driver = c10::string_view("gels");
                break;
            case 2:
                driver = c10::string_view("gelsy");
                break;
            case 3:
                driver = c10::string_view("gelsd");
                break;
            case 4:
                driver = c10::string_view("gelss");
                break;
        }

        // Calculate rcond
        c10::optional<double> rcond = c10::nullopt;
        if (rcond_byte % 4 != 0) {  // 75% chance of using rcond
            double rcond_val = static_cast<double>(rcond_byte) / 255.0 * 1e-3;
            rcond = rcond_val;
        }

        // Call torch::linalg_lstsq (note: underscore instead of ::)
        auto result = torch::linalg_lstsq(A, B, rcond, driver);

        // Extract results - lstsq returns (solution, residuals, rank, singular_values)
        torch::Tensor solution = std::get<0>(result);
        torch::Tensor residuals = std::get<1>(result);
        torch::Tensor rank = std::get<2>(result);
        torch::Tensor singular_values = std::get<3>(result);

        // Use results to prevent optimization
        if (solution.defined() && solution.numel() > 0) {
            volatile auto s = solution.sum().item<double>();
            (void)s;
        }
        if (residuals.defined() && residuals.numel() > 0) {
            volatile auto r = residuals.sum().item<double>();
            (void)r;
        }
        if (rank.defined() && rank.numel() > 0) {
            volatile auto rk = rank.item<int64_t>();
            (void)rk;
        }
        if (singular_values.defined() && singular_values.numel() > 0) {
            volatile auto sv = singular_values.sum().item<double>();
            (void)sv;
        }

        // Test with complex tensors
        if (offset < Size && Data[offset] % 3 == 0) {
            torch::Tensor A_complex = torch::randn({m, n}, torch::kComplexDouble);
            torch::Tensor B_complex = torch::randn({m, k}, torch::kComplexDouble);
            
            try {
                auto complex_result = torch::linalg_lstsq(A_complex, B_complex, rcond, driver);
                torch::Tensor complex_solution = std::get<0>(complex_result);
                if (complex_solution.defined() && complex_solution.numel() > 0) {
                    volatile auto cs = torch::real(complex_solution.sum()).item<double>();
                    (void)cs;
                }
            } catch (...) {
                // Some drivers may not support complex types
            }
        }

        // Test with batched input
        if (offset + 1 < Size && Data[offset] % 4 == 0) {
            int64_t batch = (Data[offset + 1] % 3) + 1;  // 1-3 batch size
            torch::Tensor A_batched = torch::randn({batch, m, n}, torch::kFloat64);
            torch::Tensor B_batched = torch::randn({batch, m, k}, torch::kFloat64);
            
            try {
                auto batched_result = torch::linalg_lstsq(A_batched, B_batched, rcond, driver);
                torch::Tensor batched_solution = std::get<0>(batched_result);
                if (batched_solution.defined() && batched_solution.numel() > 0) {
                    volatile auto bs = batched_solution.sum().item<double>();
                    (void)bs;
                }
            } catch (...) {
                // Batched operations may have restrictions
            }
        }

        // Test with B as a vector (1D case via 2D with k=1)
        if (offset < Size && Data[offset] % 5 == 0) {
            torch::Tensor A_vec = torch::randn({m, n}, torch::kFloat64);
            torch::Tensor B_vec = torch::randn({m}, torch::kFloat64);
            
            try {
                auto vec_result = torch::linalg_lstsq(A_vec, B_vec, rcond, driver);
                torch::Tensor vec_solution = std::get<0>(vec_result);
                if (vec_solution.defined() && vec_solution.numel() > 0) {
                    volatile auto vs = vec_solution.sum().item<double>();
                    (void)vs;
                }
            } catch (...) {
                // Vector B may have restrictions with some drivers
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