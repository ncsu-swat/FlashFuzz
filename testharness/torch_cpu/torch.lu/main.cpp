#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get

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
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Get parameters from fuzz data
        bool pivot = Data[offset++] & 0x1;
        bool get_infos = Data[offset++] & 0x1;
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point for LU decomposition
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat32);
        }
        
        // torch::lu requires at least 2D tensor
        if (A.dim() < 2) {
            return 0;
        }
        
        // Test torch::lu (the deprecated but still available API)
        // torch::lu returns std::tuple<Tensor, Tensor> (LU_data, pivots)
        // or std::tuple<Tensor, Tensor, Tensor> when get_infos=true
        try {
            if (get_infos) {
                auto result = torch::_lu_with_info(A, pivot, /*check_errors=*/false);
                auto LU_data = std::get<0>(result);
                auto pivots = std::get<1>(result);
                auto infos = std::get<2>(result);
                
                // Exercise the returned tensors
                (void)LU_data.numel();
                (void)pivots.numel();
                (void)infos.numel();
            } else {
                auto result = torch::linalg_lu_factor(A, pivot);
                auto LU_data = std::get<0>(result);
                auto pivots = std::get<1>(result);
                
                // Exercise the returned tensors
                (void)LU_data.numel();
                (void)pivots.numel();
                
                // Test lu_unpack if we have a square matrix
                if (A.size(-2) == A.size(-1)) {
                    try {
                        auto unpack_result = torch::lu_unpack(LU_data, pivots);
                        auto P = std::get<0>(unpack_result);
                        auto L = std::get<1>(unpack_result);
                        auto U = std::get<2>(unpack_result);
                        
                        (void)P.numel();
                        (void)L.numel();
                        (void)U.numel();
                    } catch (const std::exception&) {
                        // lu_unpack may fail for certain inputs
                    }
                }
                
                // Test lu_solve if square matrix
                if (A.size(-2) == A.size(-1) && A.size(-1) > 0) {
                    try {
                        // Create B tensor for solve
                        std::vector<int64_t> b_shape;
                        for (int64_t i = 0; i < A.dim() - 1; i++) {
                            b_shape.push_back(A.size(i));
                        }
                        b_shape.push_back(1);
                        
                        torch::Tensor B = torch::randn(b_shape, A.options());
                        
                        auto X = torch::lu_solve(B, LU_data, pivots);
                        (void)X.numel();
                    } catch (const std::exception&) {
                        // lu_solve may fail for singular matrices
                    }
                }
            }
        } catch (const std::exception&) {
            // LU decomposition may fail for certain inputs (singular, etc.)
        }
        
        // Also test with different dtypes
        try {
            torch::Tensor A_double = A.to(torch::kFloat64);
            auto result_double = torch::linalg_lu_factor(A_double, pivot);
            (void)std::get<0>(result_double).numel();
        } catch (const std::exception&) {
            // May fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}