#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 20) return 0;

        // Extract tensor dimensions and parameters
        auto dims = extract_tensor_dims(Data, Size, offset, 2, 4); // 2D to 4D tensors
        if (dims.empty()) return 0;

        int64_t m = std::max(1L, std::min(100L, static_cast<int64_t>(dims[dims.size()-2])));
        int64_t n = std::max(1L, std::min(100L, static_cast<int64_t>(dims[dims.size()-1])));
        
        // Update the last two dimensions
        dims[dims.size()-2] = m;
        dims[dims.size()-1] = n;

        // Extract dtype
        auto dtype = extract_dtype(Data, Size, offset);
        if (!dtype.has_value()) return 0;

        // Create input tensor A
        torch::Tensor A;
        try {
            A = create_tensor(dims, *dtype);
            if (!A.defined()) return 0;
        } catch (...) {
            return 0;
        }

        // Extract optional parameters
        bool use_q = extract_bool(Data, Size, offset);
        bool use_niter = extract_bool(Data, Size, offset);
        bool use_M = extract_bool(Data, Size, offset);

        // Extract q parameter (rank estimate)
        int64_t q = std::min(m, n); // default to min(m,n)
        if (use_q && offset < Size) {
            int64_t extracted_q = extract_int(Data, Size, offset, 1, std::min(m, n) + 5);
            q = std::max(1L, std::min(std::min(m, n) + 5, extracted_q));
        }

        // Extract niter parameter
        int64_t niter = 2; // default
        if (use_niter && offset < Size) {
            niter = extract_int(Data, Size, offset, 0, 10);
        }

        // Create optional M tensor (mean)
        torch::Tensor M;
        if (use_M) {
            try {
                auto M_dims = dims;
                M_dims[M_dims.size()-2] = 1; // M should have shape (*, 1, n)
                M = create_tensor(M_dims, *dtype);
                if (!M.defined()) {
                    use_M = false;
                }
            } catch (...) {
                use_M = false;
            }
        }

        // Test different combinations of parameters
        std::vector<std::tuple<torch::Tensor, c10::optional<int64_t>, c10::optional<int64_t>, c10::optional<torch::Tensor>>> test_cases;

        // Basic case
        test_cases.emplace_back(A, c10::nullopt, c10::nullopt, c10::nullopt);

        // With q parameter
        test_cases.emplace_back(A, q, c10::nullopt, c10::nullopt);

        // With q and niter
        test_cases.emplace_back(A, q, niter, c10::nullopt);

        // With M (if available)
        if (use_M && M.defined()) {
            test_cases.emplace_back(A, c10::nullopt, c10::nullopt, M);
            test_cases.emplace_back(A, q, c10::nullopt, M);
            test_cases.emplace_back(A, q, niter, M);
        }

        // Edge cases for q
        test_cases.emplace_back(A, 1, c10::nullopt, c10::nullopt);
        if (std::min(m, n) > 1) {
            test_cases.emplace_back(A, std::min(m, n), c10::nullopt, c10::nullopt);
        }

        // Edge cases for niter
        test_cases.emplace_back(A, c10::nullopt, 0, c10::nullopt);
        test_cases.emplace_back(A, c10::nullopt, 1, c10::nullopt);

        for (const auto& test_case : test_cases) {
            try {
                auto [test_A, test_q, test_niter, test_M] = test_case;
                
                std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> result;
                
                if (test_M.has_value()) {
                    if (test_q.has_value() && test_niter.has_value()) {
                        result = torch::svd_lowrank(test_A, test_q.value(), test_niter.value(), test_M.value());
                    } else if (test_q.has_value()) {
                        result = torch::svd_lowrank(test_A, test_q.value(), 2, test_M.value());
                    } else if (test_niter.has_value()) {
                        result = torch::svd_lowrank(test_A, c10::nullopt, test_niter.value(), test_M.value());
                    } else {
                        result = torch::svd_lowrank(test_A, c10::nullopt, 2, test_M.value());
                    }
                } else {
                    if (test_q.has_value() && test_niter.has_value()) {
                        result = torch::svd_lowrank(test_A, test_q.value(), test_niter.value());
                    } else if (test_q.has_value()) {
                        result = torch::svd_lowrank(test_A, test_q.value());
                    } else if (test_niter.has_value()) {
                        result = torch::svd_lowrank(test_A, c10::nullopt, test_niter.value());
                    } else {
                        result = torch::svd_lowrank(test_A);
                    }
                }

                // Verify result structure
                auto U = std::get<0>(result);
                auto S = std::get<1>(result);
                auto V = std::get<2>(result);

                if (!U.defined() || !S.defined() || !V.defined()) {
                    continue;
                }

                // Basic shape checks
                auto U_sizes = U.sizes();
                auto S_sizes = S.sizes();
                auto V_sizes = V.sizes();

                // Check that tensors have reasonable shapes
                if (U_sizes.size() < 2 || S_sizes.size() < 1 || V_sizes.size() < 2) {
                    continue;
                }

                // Verify no NaN or Inf values in results
                if (torch::any(torch::isnan(U)).item<bool>() || 
                    torch::any(torch::isinf(U)).item<bool>() ||
                    torch::any(torch::isnan(S)).item<bool>() || 
                    torch::any(torch::isinf(S)).item<bool>() ||
                    torch::any(torch::isnan(V)).item<bool>() || 
                    torch::any(torch::isinf(V)).item<bool>()) {
                    continue;
                }

                // Test with different tensor properties
                if (A.numel() > 0) {
                    // Test with transposed input
                    auto A_t = A.transpose(-2, -1);
                    torch::svd_lowrank(A_t);

                    // Test with contiguous tensor
                    if (!A.is_contiguous()) {
                        auto A_cont = A.contiguous();
                        torch::svd_lowrank(A_cont);
                    }
                }

            } catch (const c10::Error& e) {
                // PyTorch errors are expected for invalid inputs
                continue;
            } catch (const std::runtime_error& e) {
                // Runtime errors might be expected
                continue;
            }
        }

        // Test edge cases with special matrices
        try {
            // Zero matrix
            auto zero_A = torch::zeros_like(A);
            torch::svd_lowrank(zero_A);

            // Identity-like matrix (if square or can be made square-like)
            if (m == n && m <= 10) {
                auto eye_A = torch::eye(m, torch::dtype(*dtype));
                if (A.dim() > 2) {
                    // Expand to match batch dimensions
                    std::vector<int64_t> eye_dims(dims.begin(), dims.end()-2);
                    eye_dims.push_back(m);
                    eye_dims.push_back(n);
                    eye_A = eye_A.expand(eye_dims);
                }
                torch::svd_lowrank(eye_A);
            }

        } catch (...) {
            // Ignore errors in edge case testing
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}