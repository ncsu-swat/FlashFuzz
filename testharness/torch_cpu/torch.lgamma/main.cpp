#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply lgamma operation
        torch::Tensor result = torch::lgamma(input);
        
        // Try in-place version
        if (offset < Size) {
            // Clone to preserve original for further tests
            torch::Tensor input_copy = input.clone();
            input_copy.lgamma_();
        }
        
        // Try with out parameter
        if (offset < Size) {
            torch::Tensor out = torch::empty_like(input);
            torch::lgamma_out(out, input);
        }
        
        // Test with different dtypes if we have more data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::Tensor typed_input;
            
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        // lgamma supports complex types in some versions
                        typed_input = input.to(torch::kComplexFloat);
                        break;
                }
                torch::Tensor typed_result = torch::lgamma(typed_input);
            } catch (const std::exception&) {
                // Some dtype conversions or operations may fail
            }
        }
        
        // Test with contiguous vs non-contiguous tensors
        if (offset < Size && input.dim() >= 2) {
            try {
                // Create non-contiguous view via transpose
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor trans_result = torch::lgamma(transposed);
            } catch (const std::exception&) {
                // May fail for certain tensor configurations
            }
        }
        
        // Test with zero-dimensional tensor (scalar)
        if (offset < Size) {
            try {
                torch::Tensor scalar = torch::tensor(static_cast<float>(Data[offset++]) / 10.0f);
                torch::Tensor scalar_result = torch::lgamma(scalar);
            } catch (const std::exception&) {
                // Scalar operations might have edge cases
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}