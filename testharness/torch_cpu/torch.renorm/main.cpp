#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for renorm
        // Need at least 3 more bytes for p, dim, and maxnorm
        if (offset + 3 > Size) {
            return 0;
        }
        
        // Get p value (norm type)
        double p;
        if (Data[offset] % 4 == 0) {
            p = 1.0; // L1 norm
        } else if (Data[offset] % 4 == 1) {
            p = 2.0; // L2 norm
        } else if (Data[offset] % 4 == 2) {
            p = std::numeric_limits<double>::infinity(); // L-infinity norm
        } else {
            // Use a random p value
            uint8_t p_raw = Data[offset];
            p = static_cast<double>(p_raw) / 10.0;
        }
        offset++;
        
        // Get dimension
        int64_t dim = 0;
        if (input.dim() > 0) {
            dim = static_cast<int64_t>(Data[offset]) % input.dim();
            // Allow negative indexing
            if (Data[offset] & 0x80) {
                dim = -1 - (dim % input.dim());
            }
        }
        offset++;
        
        // Get maxnorm
        double maxnorm;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&maxnorm, Data + offset, sizeof(double));
            offset += sizeof(double);
        } else {
            // Use a default value if not enough data
            maxnorm = static_cast<double>(Data[offset]);
            offset++;
        }
        
        // Apply renorm operation
        torch::Tensor output = torch::renorm(input, p, dim, maxnorm);
        
        // Try in-place version if possible
        if (input.is_floating_point() && input.is_contiguous()) {
            try {
                torch::Tensor input_copy = input.clone();
                input_copy.renorm_(p, dim, maxnorm);
            } catch (const std::exception& e) {
                // Ignore exceptions from in-place version
            }
        }
        
        // Try different overloads and edge cases
        if (offset < Size) {
            try {
                // Try with different p values
                torch::Tensor output2 = torch::renorm(input, 0.5, dim, maxnorm);
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
            
            try {
                // Try with negative maxnorm
                torch::Tensor output3 = torch::renorm(input, p, dim, -maxnorm);
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
            
            try {
                // Try with out-of-bounds dimension
                int64_t bad_dim = input.dim() + 1;
                torch::Tensor output4 = torch::renorm(input, p, bad_dim, maxnorm);
            } catch (const std::exception& e) {
                // Ignore exceptions
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