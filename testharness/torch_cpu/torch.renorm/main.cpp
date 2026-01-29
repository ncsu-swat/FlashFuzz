#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

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
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // renorm requires floating point tensor
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // renorm requires at least 1D tensor
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // Need at least 3 more bytes for p, dim, and maxnorm
        if (offset + 3 > Size) {
            return 0;
        }
        
        // Get p value (norm type) - must be positive
        double p;
        uint8_t p_selector = Data[offset] % 5;
        if (p_selector == 0) {
            p = 1.0; // L1 norm
        } else if (p_selector == 1) {
            p = 2.0; // L2 norm
        } else if (p_selector == 2) {
            p = std::numeric_limits<double>::infinity(); // L-infinity norm
        } else if (p_selector == 3) {
            p = 0.5; // Fractional norm
        } else {
            // Use a random positive p value (0.1 to 25.6)
            uint8_t p_raw = Data[offset];
            p = (static_cast<double>(p_raw) / 10.0) + 0.1;
        }
        offset++;
        
        // Get dimension
        int64_t dim = 0;
        if (input.dim() > 0) {
            dim = static_cast<int64_t>(Data[offset]) % input.dim();
            // Allow negative indexing
            if (Data[offset] & 0x80) {
                dim = -(input.dim() - dim);
            }
        }
        offset++;
        
        // Get maxnorm - must be non-negative, avoid NaN/Inf
        double maxnorm;
        uint8_t maxnorm_raw = Data[offset];
        // Map to reasonable range [0.0, 100.0]
        maxnorm = static_cast<double>(maxnorm_raw) / 2.55;
        offset++;
        
        // Apply renorm operation
        try {
            torch::Tensor output = torch::renorm(input, p, dim, maxnorm);
        } catch (const std::exception& e) {
            // Some parameter combinations may be invalid, silently ignore
        }
        
        // Try in-place version if possible
        if (input.is_contiguous()) {
            try {
                torch::Tensor input_copy = input.clone();
                input_copy.renorm_(p, dim, maxnorm);
            } catch (const std::exception& e) {
                // Ignore exceptions from in-place version
            }
        }
        
        // Try different overloads and edge cases
        if (offset < Size) {
            // Try with different p values
            try {
                torch::Tensor output2 = torch::renorm(input, 0.5, dim, maxnorm);
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
            
            try {
                torch::Tensor output3 = torch::renorm(input, 1.0, dim, maxnorm);
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
            
            try {
                torch::Tensor output4 = torch::renorm(input, 2.0, dim, maxnorm);
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
            
            // Try with zero maxnorm
            try {
                torch::Tensor output5 = torch::renorm(input, p, dim, 0.0);
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
            
            // Try with very large maxnorm
            try {
                torch::Tensor output6 = torch::renorm(input, p, dim, 1e10);
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
            
            // Try with different dimensions
            for (int64_t d = 0; d < input.dim() && d < 3; d++) {
                try {
                    torch::Tensor output_dim = torch::renorm(input, p, d, maxnorm);
                } catch (const std::exception& e) {
                    // Ignore exceptions
                }
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