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
        
        // Convert to float for arccosh (requires floating point)
        torch::Tensor float_input = input.to(torch::kFloat32);
        
        // Apply arccosh operation
        torch::Tensor result = torch::arccosh(float_input);
        
        // Try in-place version
        try {
            torch::Tensor input_copy = float_input.clone();
            input_copy.arccosh_();
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Try with different dtype if there's more data
        if (offset + 1 < Size) {
            try {
                uint8_t dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                // Only use floating point dtypes for arccosh
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || 
                    dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
                    torch::Tensor input_cast = float_input.to(dtype);
                    torch::Tensor result_with_dtype = torch::arccosh(input_cast);
                }
            } catch (...) {
                // Silently catch dtype conversion failures
            }
        }
        
        // Try out-of-place with named output tensor
        try {
            torch::Tensor output = torch::empty_like(float_input);
            torch::arccosh_out(output, float_input);
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Test with values in valid domain (>= 1) to ensure proper coverage
        if (offset + 4 < Size) {
            try {
                // Create a tensor with values shifted to valid domain
                torch::Tensor valid_input = torch::abs(float_input) + 1.0;
                torch::Tensor valid_result = torch::arccosh(valid_input);
            } catch (...) {
                // Silently catch expected failures
            }
        }
        
        // Test with different tensor shapes
        if (offset + 2 < Size) {
            try {
                int dim1 = (Data[offset++] % 8) + 1;
                int dim2 = (Data[offset++] % 8) + 1;
                torch::Tensor shaped_input = torch::randn({dim1, dim2});
                torch::Tensor shaped_result = torch::arccosh(shaped_input);
            } catch (...) {
                // Silently catch shape-related failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}