#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for tensor creation
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor x
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input tensor n (should be integer type)
        torch::Tensor n;
        if (offset < Size) {
            n = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert n to integer type if needed
            if (!n.dtype().is_integral()) {
                n = n.to(torch::kInt64);
            }
        } else {
            // If we don't have enough data for a second tensor, create a scalar n
            n = torch::tensor(1, torch::kInt64);
        }
        
        // Call the Chebyshev polynomial function
        torch::Tensor result = torch::special::chebyshev_polynomial_u(x, n);
        
        // Try different variants of the function
        if (Size % 3 == 0 && offset < Size) {
            // Try with scalar n
            int64_t scalar_n = Data[offset] % 10; // Use a small integer value
            torch::Tensor result2 = torch::special::chebyshev_polynomial_u(x, scalar_n);
        }
        
        if (Size % 3 == 1 && x.dim() > 0) {
            // Try broadcasting with n having different shape
            std::vector<int64_t> new_shape;
            for (int i = 0; i < x.dim() - 1; i++) {
                new_shape.push_back(1);
            }
            new_shape.push_back(x.size(-1));
            
            torch::Tensor reshaped_n = n.reshape(new_shape);
            torch::Tensor result3 = torch::special::chebyshev_polynomial_u(x, reshaped_n);
        }
        
        // Try with extreme values for n
        if (Size % 3 == 2) {
            torch::Tensor extreme_n;
            if (offset < Size) {
                uint8_t selector = Data[offset++];
                if (selector % 4 == 0) {
                    extreme_n = torch::tensor(0, torch::kInt64); // n = 0
                } else if (selector % 4 == 1) {
                    extreme_n = torch::tensor(-1, torch::kInt64); // n = -1
                } else if (selector % 4 == 2) {
                    extreme_n = torch::tensor(100, torch::kInt64); // large n
                } else {
                    extreme_n = torch::tensor(-100, torch::kInt64); // large negative n
                }
                torch::Tensor result4 = torch::special::chebyshev_polynomial_u(x, extreme_n);
            }
        }
        
        // Try with extreme values for x
        if (Size % 5 == 0 && offset < Size) {
            torch::Tensor extreme_x;
            uint8_t selector = Data[offset++];
            if (selector % 3 == 0) {
                extreme_x = torch::full_like(x, std::numeric_limits<float>::infinity());
            } else if (selector % 3 == 1) {
                extreme_x = torch::full_like(x, -std::numeric_limits<float>::infinity());
            } else {
                extreme_x = torch::full_like(x, std::numeric_limits<float>::quiet_NaN());
            }
            
            torch::Tensor result5 = torch::special::chebyshev_polynomial_u(extreme_x, n);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}