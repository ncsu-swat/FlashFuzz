#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <limits>         // For numeric_limits

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
            // Check if dtype is not an integral type by comparing with known integral types
            auto dtype = n.scalar_type();
            if (dtype != torch::kInt8 && dtype != torch::kInt16 && 
                dtype != torch::kInt32 && dtype != torch::kInt64 &&
                dtype != torch::kUInt8) {
                n = n.to(torch::kInt64);
            }
        } else {
            // If we don't have enough data for a second tensor, create a scalar n
            n = torch::tensor(1, torch::kInt64);
        }
        
        // Call the Chebyshev polynomial function
        try {
            torch::Tensor result = torch::special::chebyshev_polynomial_u(x, n);
        } catch (...) {
            // Silently handle expected failures (shape mismatch, etc.)
        }
        
        // Try different variants of the function
        if (Size % 3 == 0 && offset < Size) {
            // Try with scalar n
            int64_t scalar_n = static_cast<int64_t>(Data[offset] % 10); // Use a small integer value
            try {
                torch::Tensor result2 = torch::special::chebyshev_polynomial_u(x, scalar_n);
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        if (Size % 3 == 1 && x.dim() > 0 && x.numel() > 0) {
            // Try broadcasting with n having different shape
            try {
                std::vector<int64_t> new_shape;
                for (int i = 0; i < x.dim() - 1; i++) {
                    new_shape.push_back(1);
                }
                new_shape.push_back(x.size(-1));
                
                // Only reshape if n has enough elements
                if (n.numel() >= x.size(-1)) {
                    torch::Tensor flat_n = n.flatten().slice(0, 0, x.size(-1));
                    torch::Tensor reshaped_n = flat_n.reshape(new_shape);
                    torch::Tensor result3 = torch::special::chebyshev_polynomial_u(x, reshaped_n);
                }
            } catch (...) {
                // Silently handle reshape/broadcast failures
            }
        }
        
        // Try with extreme values for n
        if (Size % 3 == 2) {
            torch::Tensor extreme_n;
            size_t local_offset = offset;
            if (local_offset < Size) {
                uint8_t selector = Data[local_offset++];
                if (selector % 4 == 0) {
                    extreme_n = torch::tensor(0, torch::kInt64); // n = 0
                } else if (selector % 4 == 1) {
                    extreme_n = torch::tensor(-1, torch::kInt64); // n = -1
                } else if (selector % 4 == 2) {
                    extreme_n = torch::tensor(100, torch::kInt64); // large n
                } else {
                    extreme_n = torch::tensor(-100, torch::kInt64); // large negative n
                }
                try {
                    torch::Tensor result4 = torch::special::chebyshev_polynomial_u(x, extreme_n);
                } catch (...) {
                    // Silently handle expected failures
                }
            }
        }
        
        // Try with extreme values for x
        if (Size % 5 == 0 && offset < Size) {
            torch::Tensor extreme_x;
            uint8_t selector = Data[offset];
            if (selector % 3 == 0) {
                extreme_x = torch::full_like(x, std::numeric_limits<float>::infinity());
            } else if (selector % 3 == 1) {
                extreme_x = torch::full_like(x, -std::numeric_limits<float>::infinity());
            } else {
                extreme_x = torch::full_like(x, std::numeric_limits<float>::quiet_NaN());
            }
            
            try {
                torch::Tensor result5 = torch::special::chebyshev_polynomial_u(extreme_x, n);
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Try with double precision x
        if (Size % 7 == 0) {
            try {
                torch::Tensor x_double = x.to(torch::kFloat64);
                torch::Tensor result6 = torch::special::chebyshev_polynomial_u(x_double, n);
            } catch (...) {
                // Silently handle expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}