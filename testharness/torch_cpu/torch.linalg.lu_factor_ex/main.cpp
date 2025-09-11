#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for lu_factor_ex
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 2 dimensions for lu_factor_ex
        if (A.dim() < 2) {
            // Reshape to 2D if needed
            if (A.dim() == 0) {
                A = A.reshape({1, 1});
            } else if (A.dim() == 1) {
                int64_t size = A.size(0);
                A = A.reshape({1, size});
            }
        }
        
        // Parse pivot flag from input data
        bool pivot = true;
        if (offset < Size) {
            pivot = Data[offset++] & 0x1;
        }
        
        // Parse check_errors flag from input data
        bool check_errors = true;
        if (offset < Size) {
            check_errors = Data[offset++] & 0x1;
        }
        
        // Apply lu_factor_ex operation
        auto result = torch::lu_factor_ex(A, pivot, check_errors);
        
        // Access the results to ensure they're computed
        auto LU = std::get<0>(result);
        auto pivots = std::get<1>(result);
        auto info = std::get<2>(result);
        
        // Perform some operations on the results to ensure they're valid
        if (LU.numel() > 0) {
            auto sum = LU.sum();
        }
        
        if (pivots.numel() > 0) {
            auto max_pivot = pivots.max();
        }
        
        if (info.numel() > 0) {
            auto max_info = info.max();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
