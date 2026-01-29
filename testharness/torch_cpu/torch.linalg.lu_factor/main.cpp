#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>
#include <cmath>

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
        
        // Create input tensor for lu_factor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // lu_factor requires at least a 2D tensor with last two dimensions forming a matrix
        auto dims = A.sizes().vec();
        
        // Handle 0-dim or 1-dim tensors
        if (dims.size() <= 1) {
            int64_t total_elements = A.numel();
            if (total_elements == 0) {
                return 0;
            }
            int64_t matrix_size = std::max(static_cast<int64_t>(1), 
                static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements))));
            
            // Limit matrix size to avoid excessive computation
            matrix_size = std::min(matrix_size, static_cast<int64_t>(64));
            
            int64_t needed = matrix_size * matrix_size;
            if (needed > total_elements) {
                A = torch::constant_pad_nd(A.reshape({-1}), {0, needed - total_elements}, 0);
            } else {
                A = A.reshape({-1}).slice(0, 0, needed);
            }
            A = A.reshape({matrix_size, matrix_size});
        }
        else {
            // For 2+ dimensional tensors, ensure the last two dimensions form a valid matrix
            auto last_dim = dims.back();
            auto second_last_dim = dims[dims.size() - 2];
            
            // Limit dimensions to avoid excessive computation
            int64_t max_dim = 64;
            last_dim = std::min(last_dim, max_dim);
            second_last_dim = std::min(second_last_dim, max_dim);
            
            if (last_dim != second_last_dim) {
                int64_t square_size = std::min(last_dim, second_last_dim);
                if (square_size == 0) square_size = 1;
                
                // Use narrow to get a square submatrix
                A = A.narrow(-2, 0, square_size).narrow(-1, 0, square_size);
            }
        }
        
        // Ensure tensor is floating point type for lu_factor
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat32);
        }
        
        // Ensure tensor is contiguous
        A = A.contiguous();
        
        // Check for valid dimensions
        if (A.dim() < 2 || A.size(-1) == 0 || A.size(-2) == 0) {
            return 0;
        }
        
        // Apply torch::linalg_lu_factor operation (C++ API uses flat namespace)
        auto result = torch::linalg_lu_factor(A);
        
        // Unpack the result
        auto& LU = std::get<0>(result);
        auto& pivots = std::get<1>(result);
        
        // Perform basic operations on the results to ensure they're valid
        auto LU_sum = LU.sum();
        auto pivots_sum = pivots.sum();
        
        // Access the values to prevent optimization
        volatile double lu_val = LU_sum.item<double>();
        volatile int64_t piv_val = pivots_sum.item<int64_t>();
        (void)lu_val;
        (void)piv_val;
        
        // Test with pivot=False option as well (exercises different code path)
        if (Size > 0 && (Data[0] & 1)) {
            try {
                auto result_no_pivot = torch::linalg_lu_factor(A, /*pivot=*/false);
                auto& LU2 = std::get<0>(result_no_pivot);
                volatile double lu2_val = LU2.sum().item<double>();
                (void)lu2_val;
            } catch (const std::exception& e) {
                // lu_factor without pivoting may fail for some matrices
            }
        }
        
        // Test lu_factor_ex for more coverage (returns info tensor as well)
        if (Size > 1 && (Data[1] & 1)) {
            try {
                auto result_ex = torch::linalg_lu_factor_ex(A);
                auto& LU3 = std::get<0>(result_ex);
                auto& pivots3 = std::get<1>(result_ex);
                auto& info = std::get<2>(result_ex);
                
                volatile double lu3_val = LU3.sum().item<double>();
                volatile int64_t piv3_val = pivots3.sum().item<int64_t>();
                volatile int64_t info_val = info.sum().item<int64_t>();
                (void)lu3_val;
                (void)piv3_val;
                (void)info_val;
            } catch (const std::exception& e) {
                // Expected for some edge cases
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