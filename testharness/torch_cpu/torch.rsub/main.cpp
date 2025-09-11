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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create scalar or tensor for the "other" parameter
        torch::Tensor other;
        if (offset < Size) {
            // Create another tensor for "other" parameter
            other = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Use a scalar value if we don't have enough data
            other = torch::tensor(2.0);
        }
        
        // Get alpha value from remaining data if available
        double alpha = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Apply rsub operation in different ways to maximize coverage
        
        // 1. Basic rsub: other - input
        torch::Tensor result1 = torch::rsub(input, other);
        
        // 2. rsub with alpha: other - input * alpha
        torch::Tensor result2 = torch::rsub(input, other, alpha);
        
        // 3. rsub with scalar
        torch::Tensor result3;
        if (offset < Size) {
            double scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            result3 = torch::rsub(input, scalar_value);
        }
        
        // 4. rsub with scalar and alpha
        torch::Tensor result4;
        if (offset + sizeof(double) <= Size) {
            double scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            result4 = torch::rsub(input, scalar_value, alpha);
        }
        
        // 5. In-place version using assignment
        if (other.sizes() == input.sizes() && other.dtype() == input.dtype()) {
            try {
                torch::Tensor input_copy = input.clone();
                input_copy = torch::rsub(input_copy, other, alpha);
            } catch (const std::exception &) {
                // Ignore exceptions from in-place operations
            }
        }
        
        // 6. Test with extreme values for alpha
        if (offset + sizeof(double) <= Size) {
            double extreme_alpha;
            std::memcpy(&extreme_alpha, Data + offset, sizeof(double));
            // Make it potentially very large or very small
            extreme_alpha = std::pow(10.0, extreme_alpha);
            try {
                torch::Tensor result_extreme = torch::rsub(input, other, extreme_alpha);
            } catch (const std::exception &) {
                // Ignore exceptions from extreme values
            }
        }
        
        // 7. Test with scalar tensors
        try {
            if (input.numel() == 1) {
                torch::Tensor scalar_result = torch::rsub(input, 5.0);
            }
        } catch (const std::exception &) {
            // Ignore exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
