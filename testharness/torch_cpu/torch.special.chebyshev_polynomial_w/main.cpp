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
        
        // Need at least some data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for n (order of the polynomial)
        torch::Tensor n = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input tensor for x (points at which to evaluate)
        if (offset < Size) {
            torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply the Chebyshev polynomial of the third kind (W_n(x))
            torch::Tensor result = torch::special::chebyshev_polynomial_w(x, n);
            
            // Try with default x=1
            torch::Tensor x_default = torch::tensor(1.0);
            torch::Tensor result_default = torch::special::chebyshev_polynomial_w(x_default, n);
            
            // Try with scalar inputs
            if (offset + 2 < Size) {
                // Extract scalar values from the data
                double n_scalar = static_cast<double>(Data[offset++]);
                double x_scalar = static_cast<double>(Data[offset++]);
                
                // Create scalar tensors
                torch::Tensor n_tensor = torch::tensor(n_scalar);
                torch::Tensor x_tensor = torch::tensor(x_scalar);
                
                // Apply the operation with scalar inputs
                torch::Tensor result_scalar = torch::special::chebyshev_polynomial_w(x_tensor, n_tensor);
            }
            
            // Try with empty tensors
            if (offset < Size) {
                try {
                    torch::Tensor empty_n = torch::empty({0});
                    torch::Tensor empty_x = torch::empty({0});
                    torch::Tensor result_empty = torch::special::chebyshev_polynomial_w(empty_x, empty_n);
                } catch (const std::exception&) {
                    // Expected to potentially fail, but we want to test it
                }
            }
            
            // Try with tensors of different shapes
            if (offset < Size) {
                try {
                    torch::Tensor n_vec = fuzzer_utils::createTensor(Data, Size, offset);
                    torch::Tensor x_scalar = torch::tensor(1.0);
                    torch::Tensor result_mixed = torch::special::chebyshev_polynomial_w(x_scalar, n_vec);
                } catch (const std::exception&) {
                    // Expected to potentially fail with shape mismatch
                }
            }
            
            // Try with extreme values
            if (offset < Size) {
                try {
                    torch::Tensor extreme_n = torch::tensor(1e10);
                    torch::Tensor extreme_x = torch::tensor(1e10);
                    torch::Tensor result_extreme = torch::special::chebyshev_polynomial_w(extreme_x, extreme_n);
                } catch (const std::exception&) {
                    // Expected to potentially fail with numerical issues
                }
            }
            
            // Try with negative values
            if (offset < Size) {
                try {
                    torch::Tensor neg_n = torch::tensor(-10.0);
                    torch::Tensor neg_x = torch::tensor(-0.5);
                    torch::Tensor result_neg = torch::special::chebyshev_polynomial_w(neg_x, neg_n);
                } catch (const std::exception&) {
                    // Expected to potentially fail with domain issues
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
