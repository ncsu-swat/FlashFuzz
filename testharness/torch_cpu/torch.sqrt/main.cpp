#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply sqrt operation
        torch::Tensor result = torch::sqrt(input);
        
        // Try inplace version with a clone (sqrt_ requires floating point)
        try {
            torch::Tensor input_copy = input.clone();
            if (input_copy.is_floating_point() || input_copy.is_complex()) {
                input_copy.sqrt_();
            }
        } catch (const std::exception&) {
            // Ignore - inplace may fail for certain tensor types
        }
        
        // Try with out parameter
        try {
            torch::Tensor out = torch::empty_like(input);
            torch::sqrt_out(out, input);
        } catch (const std::exception&) {
            // Ignore - out variant may fail for certain tensor types
        }
        
        // Try with complex tensors specifically
        try {
            if (input.is_complex()) {
                torch::Tensor complex_result = torch::sqrt(input);
            } else if (input.is_floating_point()) {
                // Create a complex tensor from floating point input
                torch::Tensor complex_input = torch::complex(input, input);
                torch::Tensor complex_result = torch::sqrt(complex_input);
            }
        } catch (const std::exception&) {
            // Ignore exceptions from complex tensor operations
        }
        
        // Try with negative values to test NaN behavior
        try {
            if (input.is_floating_point()) {
                torch::Tensor neg_input = -torch::abs(input) - 1.0;
                torch::Tensor neg_result = torch::sqrt(neg_input);
                // Result should contain NaN values
            }
        } catch (const std::exception&) {
            // Ignore exceptions from negative inputs
        }
        
        // Test with different tensor options if we have extra data
        if (offset < Size && Size - offset >= 1) {
            try {
                uint8_t dtype_selector = Data[offset] % 4;
                offset++;
                
                torch::Tensor typed_input;
                switch (dtype_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input.to(torch::kComplexFloat);
                        break;
                    case 3:
                        typed_input = input.to(torch::kComplexDouble);
                        break;
                }
                torch::Tensor typed_result = torch::sqrt(typed_input);
            } catch (const std::exception&) {
                // Ignore type conversion failures
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