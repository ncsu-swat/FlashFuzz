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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a square matrix for slogdet
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // slogdet requires a square matrix (n x n)
        // If tensor is not 2D, reshape it to a square matrix if possible
        if (input.dim() != 2 || input.size(0) != input.size(1)) {
            int64_t total_elements = input.numel();
            int64_t matrix_size = static_cast<int64_t>(std::sqrt(total_elements));
            
            // Ensure we have enough elements for at least a 1x1 matrix
            if (matrix_size > 0) {
                // Reshape to square matrix
                input = input.reshape({matrix_size, matrix_size});
            } else {
                // Create a minimal 1x1 matrix if we don't have enough elements
                input = torch::ones({1, 1}, input.options());
            }
        }
        
        // Apply slogdet operation
        auto result = torch::slogdet(input);
        
        // Unpack the result (sign, logabsdet)
        auto sign = std::get<0>(result);
        auto logabsdet = std::get<1>(result);
        
        // Try to use the results to ensure they're not optimized away
        if (offset < Size) {
            // Create a simple operation using the results
            auto combined = sign * torch::exp(logabsdet);
            
            // Perform another operation to test edge cases
            if (offset + 1 < Size) {
                uint8_t op_selector = Data[offset++];
                
                // Different operations based on the selector
                switch (op_selector % 4) {
                    case 0:
                        // Test inverse if determinant is non-zero
                        if (!torch::any(sign == 0).item<bool>()) {
                            try {
                                auto inv = torch::inverse(input);
                                auto inv_det = torch::slogdet(inv);
                            } catch (...) {
                                // Ignore errors from inverse calculation
                            }
                        }
                        break;
                    case 1:
                        // Test with transposed matrix
                        try {
                            auto transposed = input.transpose(0, 1);
                            auto trans_det = torch::slogdet(transposed);
                        } catch (...) {
                            // Ignore errors
                        }
                        break;
                    case 2:
                        // Test with scaled matrix
                        try {
                            auto scaled = input * 2.0;
                            auto scaled_det = torch::slogdet(scaled);
                        } catch (...) {
                            // Ignore errors
                        }
                        break;
                    case 3:
                        // Test with matrix power if square
                        try {
                            auto power = torch::matrix_power(input, 2);
                            auto power_det = torch::slogdet(power);
                        } catch (...) {
                            // Ignore errors
                        }
                        break;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
