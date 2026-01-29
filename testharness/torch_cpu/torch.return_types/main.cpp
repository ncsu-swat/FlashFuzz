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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // 1. Test torch::max with dim - returns (values, indices)
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            try {
                int64_t dim = tensor.dim() > 0 ? 0 : -1;
                auto max_result = torch::max(tensor, dim);
                torch::Tensor values = std::get<0>(max_result);
                torch::Tensor indices = std::get<1>(max_result);
                (void)values;
                (void)indices;
            } catch (const std::exception&) {
            }
        }
        
        // 2. Test torch::min with dim - returns (values, indices)
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            try {
                int64_t dim = 0;
                auto min_result = torch::min(tensor, dim);
                torch::Tensor values = std::get<0>(min_result);
                torch::Tensor indices = std::get<1>(min_result);
                (void)values;
                (void)indices;
            } catch (const std::exception&) {
            }
        }
        
        // 3. Test torch::sort - returns (sorted, indices)
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            try {
                int64_t dim = 0;
                bool descending = (offset < Size && Data[offset++] % 2 == 0);
                auto sort_result = torch::sort(tensor, dim, descending);
                torch::Tensor values = std::get<0>(sort_result);
                torch::Tensor indices = std::get<1>(sort_result);
                (void)values;
                (void)indices;
            } catch (const std::exception&) {
            }
        }
        
        // 4. Test torch::topk - returns (values, indices)
        if (tensor.dim() > 0 && tensor.size(0) > 0) {
            try {
                int64_t k = 1;
                if (tensor.size(0) > 1 && offset < Size) {
                    k = (Data[offset++] % tensor.size(0)) + 1;
                }
                int64_t dim = 0;
                bool largest = (offset < Size && Data[offset++] % 2 == 0);
                bool sorted = (offset < Size && Data[offset++] % 2 == 0);
                
                auto topk_result = torch::topk(tensor, k, dim, largest, sorted);
                torch::Tensor values = std::get<0>(topk_result);
                torch::Tensor indices = std::get<1>(topk_result);
                (void)values;
                (void)indices;
            } catch (const std::exception&) {
            }
        }
        
        // 5. Test torch::svd - returns (U, S, V)
        if (tensor.dim() >= 2) {
            try {
                auto svd_result = torch::svd(tensor.to(torch::kFloat));
                torch::Tensor U = std::get<0>(svd_result);
                torch::Tensor S = std::get<1>(svd_result);
                torch::Tensor V = std::get<2>(svd_result);
                (void)U;
                (void)S;
                (void)V;
            } catch (const std::exception&) {
            }
        }
        
        // 6. Test torch::mode - returns (values, indices)
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            try {
                int64_t dim = 0;
                auto mode_result = torch::mode(tensor, dim);
                torch::Tensor values = std::get<0>(mode_result);
                torch::Tensor indices = std::get<1>(mode_result);
                (void)values;
                (void)indices;
            } catch (const std::exception&) {
            }
        }
        
        // 7. Test torch::median with dim - returns (values, indices)
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            try {
                int64_t dim = 0;
                auto median_result = torch::median(tensor, dim);
                torch::Tensor values = std::get<0>(median_result);
                torch::Tensor indices = std::get<1>(median_result);
                (void)values;
                (void)indices;
            } catch (const std::exception&) {
            }
        }
        
        // 8. Test torch::kthvalue - returns (values, indices)
        if (tensor.dim() > 0 && tensor.size(0) > 0) {
            try {
                int64_t k = 1;
                if (tensor.size(0) > 1 && offset < Size) {
                    k = (Data[offset++] % tensor.size(0)) + 1;
                }
                int64_t dim = 0;
                bool keepdim = (offset < Size && Data[offset++] % 2 == 0);
                
                auto kthvalue_result = torch::kthvalue(tensor, k, dim, keepdim);
                torch::Tensor values = std::get<0>(kthvalue_result);
                torch::Tensor indices = std::get<1>(kthvalue_result);
                (void)values;
                (void)indices;
            } catch (const std::exception&) {
            }
        }
        
        // 9. Test torch::linalg_qr - returns (Q, R)
        if (tensor.dim() >= 2) {
            try {
                auto qr_result = torch::linalg_qr(tensor.to(torch::kFloat));
                torch::Tensor Q = std::get<0>(qr_result);
                torch::Tensor R = std::get<1>(qr_result);
                (void)Q;
                (void)R;
            } catch (const std::exception&) {
            }
        }
        
        // 10. Test torch::lu_unpack with torch::_lu_with_info for LU decomposition
        if (tensor.dim() >= 2 && tensor.size(-1) == tensor.size(-2)) {
            try {
                auto lu_result = torch::_lu_with_info(tensor.to(torch::kFloat));
                torch::Tensor LU = std::get<0>(lu_result);
                torch::Tensor pivots = std::get<1>(lu_result);
                (void)LU;
                (void)pivots;
            } catch (const std::exception&) {
            }
        }
        
        // 11. Test torch::cummax - returns (values, indices)
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            try {
                int64_t dim = 0;
                auto cummax_result = torch::cummax(tensor, dim);
                torch::Tensor values = std::get<0>(cummax_result);
                torch::Tensor indices = std::get<1>(cummax_result);
                (void)values;
                (void)indices;
            } catch (const std::exception&) {
            }
        }
        
        // 12. Test torch::cummin - returns (values, indices)
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            try {
                int64_t dim = 0;
                auto cummin_result = torch::cummin(tensor, dim);
                torch::Tensor values = std::get<0>(cummin_result);
                torch::Tensor indices = std::get<1>(cummin_result);
                (void)values;
                (void)indices;
            } catch (const std::exception&) {
            }
        }
        
        // 13. Test torch::frexp - returns (mantissa, exponent)
        try {
            auto frexp_result = torch::frexp(tensor.to(torch::kFloat));
            torch::Tensor mantissa = std::get<0>(frexp_result);
            torch::Tensor exponent = std::get<1>(frexp_result);
            (void)mantissa;
            (void)exponent;
        } catch (const std::exception&) {
        }
        
        // 14. Test torch::aminmax - returns (min, max)
        if (tensor.numel() > 0) {
            try {
                auto aminmax_result = torch::aminmax(tensor);
                torch::Tensor min_val = std::get<0>(aminmax_result);
                torch::Tensor max_val = std::get<1>(aminmax_result);
                (void)min_val;
                (void)max_val;
            } catch (const std::exception&) {
            }
        }
        
        // 15. Test torch::geqrf - returns (a, tau) for QR factorization
        if (tensor.dim() >= 2) {
            try {
                auto geqrf_result = torch::geqrf(tensor.to(torch::kFloat));
                torch::Tensor a = std::get<0>(geqrf_result);
                torch::Tensor tau = std::get<1>(geqrf_result);
                (void)a;
                (void)tau;
            } catch (const std::exception&) {
            }
        }
        
        // 16. Test torch::linalg_eig - returns (eigenvalues, eigenvectors)
        if (tensor.dim() >= 2 && tensor.size(-1) == tensor.size(-2)) {
            try {
                auto eig_result = torch::linalg_eig(tensor.to(torch::kFloat));
                torch::Tensor eigenvalues = std::get<0>(eig_result);
                torch::Tensor eigenvectors = std::get<1>(eig_result);
                (void)eigenvalues;
                (void)eigenvectors;
            } catch (const std::exception&) {
            }
        }
        
        // 17. Test torch::linalg_slogdet - returns (sign, logabsdet)
        if (tensor.dim() >= 2 && tensor.size(-1) == tensor.size(-2)) {
            try {
                auto slogdet_result = torch::linalg_slogdet(tensor.to(torch::kFloat));
                torch::Tensor sign = std::get<0>(slogdet_result);
                torch::Tensor logabsdet = std::get<1>(slogdet_result);
                (void)sign;
                (void)logabsdet;
            } catch (const std::exception&) {
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