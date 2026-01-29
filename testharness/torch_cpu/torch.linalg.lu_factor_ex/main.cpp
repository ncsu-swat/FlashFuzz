#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <tuple>          // For std::get with lu_factor_ex result

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
        
        // Parse matrix dimensions from input data
        // Keep dimensions small to avoid memory issues
        int64_t m = (Data[offset++] % 16) + 1;  // rows: 1-16
        int64_t n = (Data[offset++] % 16) + 1;  // cols: 1-16
        
        // Parse pivot flag from input data
        bool pivot = Data[offset++] & 0x1;
        
        // Parse check_errors flag from input data
        bool check_errors = Data[offset++] & 0x1;
        
        // Create input tensor for lu_factor_ex
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point (required for LU factorization)
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat32);
        }
        
        // Reshape to proper 2D matrix dimensions
        int64_t total_elements = A.numel();
        if (total_elements == 0) {
            // Create a minimal matrix if tensor is empty
            A = torch::randn({m, n});
        } else {
            // Reshape to 2D, adjusting dimensions based on available elements
            int64_t needed = m * n;
            if (total_elements < needed) {
                // Reduce dimensions to fit available data
                m = std::max(int64_t(1), static_cast<int64_t>(std::sqrt(total_elements)));
                n = std::max(int64_t(1), total_elements / m);
            }
            A = A.flatten().slice(0, 0, m * n).reshape({m, n}).to(torch::kFloat32);
        }
        
        // Apply lu_factor_ex operation using correct function name (not nested namespace)
        auto result = torch::linalg_lu_factor_ex(A, pivot, check_errors);
        
        // Access the results to ensure they're computed
        auto LU = std::get<0>(result);
        auto pivots = std::get<1>(result);
        auto info = std::get<2>(result);
        
        // Perform some operations on the results to ensure they're valid
        if (LU.numel() > 0) {
            auto sum = LU.sum();
            (void)sum;  // Prevent unused variable warning
        }
        
        if (pivots.numel() > 0) {
            auto max_pivot = pivots.max();
            (void)max_pivot;
        }
        
        if (info.numel() > 0) {
            auto max_info = info.max();
            (void)max_info;
        }
        
        // Also test with batched input (3D tensor)
        if (Size > offset + 1) {
            int64_t batch_size = (Data[offset] % 4) + 1;  // 1-4 batches
            try {
                torch::Tensor A_batched = torch::randn({batch_size, m, n});
                
                // Fill with some fuzzed data
                if (offset + 4 < Size) {
                    float scale = static_cast<float>(Data[offset + 1]) / 25.5f;
                    A_batched = A_batched * scale;
                }
                
                auto result_batched = torch::linalg_lu_factor_ex(A_batched, pivot, check_errors);
                auto LU_batched = std::get<0>(result_batched);
                (void)LU_batched;
            }
            catch (const std::exception &) {
                // Silently ignore batched test failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}