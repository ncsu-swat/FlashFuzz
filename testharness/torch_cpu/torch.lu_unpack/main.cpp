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
        
        // Need sufficient bytes for tensor creation and flags
        if (Size < 8) {
            return 0;
        }
        
        // Create the LU factorization tensor - must be at least 2D
        torch::Tensor LU_raw = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure LU tensor is at least 2D and floating point
        torch::Tensor LU;
        if (LU_raw.dim() < 2) {
            // Make it 2D by reshaping or expanding
            int64_t size = std::max(LU_raw.numel(), (int64_t)4);
            int64_t side = (int64_t)std::sqrt((double)size);
            if (side < 2) side = 2;
            LU = torch::randn({side, side});
        } else {
            LU = LU_raw;
        }
        
        // Ensure floating point type for LU decomposition
        if (!LU.is_floating_point()) {
            LU = LU.to(torch::kFloat);
        }
        
        // Get dimensions
        int64_t m = LU.size(-2);
        int64_t n = LU.size(-1);
        int64_t k = std::min(m, n);
        
        // Create properly shaped pivot tensor
        // Pivots should be 1-indexed integers with shape (..., k)
        std::vector<int64_t> pivot_shape;
        for (int64_t i = 0; i < LU.dim() - 2; i++) {
            pivot_shape.push_back(LU.size(i));
        }
        pivot_shape.push_back(k);
        
        // Create identity permutation pivots (1-indexed)
        torch::Tensor pivots = torch::arange(1, k + 1, torch::kInt32);
        if (pivot_shape.size() > 1) {
            // Broadcast to batch dimensions
            std::vector<int64_t> expand_shape = pivot_shape;
            pivots = pivots.expand(expand_shape).contiguous();
        }
        
        // Get unpack flags from fuzzer data
        bool unpack_data = true;
        bool unpack_pivots = true;
        if (offset < Size) {
            unpack_data = Data[offset++] & 0x1;
        }
        if (offset < Size) {
            unpack_pivots = Data[offset++] & 0x1;
        }
        
        // Inner try-catch for expected failures (shape mismatches, etc.)
        try {
            // Call lu_unpack
            auto result = torch::lu_unpack(LU, pivots, unpack_data, unpack_pivots);
            
            torch::Tensor P = std::get<0>(result);
            torch::Tensor L = std::get<1>(result);
            torch::Tensor U = std::get<2>(result);
            
            // Exercise the results
            if (unpack_pivots && P.numel() > 0) {
                volatile float p_sum = P.sum().item<float>();
                (void)p_sum;
            }
            
            if (unpack_data && L.numel() > 0 && U.numel() > 0) {
                // Verify L and U have expected properties
                volatile float l_sum = L.sum().item<float>();
                volatile float u_sum = U.sum().item<float>();
                (void)l_sum;
                (void)u_sum;
            }
        }
        catch (const c10::Error&) {
            // Expected failures from invalid shapes/types - silently ignore
        }
        catch (const std::runtime_error&) {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}