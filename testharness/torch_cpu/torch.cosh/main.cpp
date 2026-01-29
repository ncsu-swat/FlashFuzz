#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        
        // Apply cosh operation
        torch::Tensor result = torch::cosh(input);
        
        // Try in-place version
        {
            torch::Tensor input_copy = input.clone();
            input_copy.cosh_();
        }
        
        // Try with different dtypes if there's more data
        if (offset < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Test with float dtype
            if (option_byte % 2 == 0) {
                try {
                    torch::Tensor float_input = input.to(torch::kFloat32);
                    torch::Tensor result_float = torch::cosh(float_input);
                } catch (...) {
                    // Silently handle conversion failures
                }
            }
            
            // Test with double dtype
            if (option_byte % 3 == 0) {
                try {
                    torch::Tensor double_input = input.to(torch::kFloat64);
                    torch::Tensor result_double = torch::cosh(double_input);
                } catch (...) {
                    // Silently handle conversion failures
                }
            }
        }
        
        // Test with output tensor
        if (offset < Size) {
            try {
                torch::Tensor out_tensor = torch::empty_like(input.to(torch::kFloat32));
                torch::cosh_out(out_tensor, input.to(torch::kFloat32));
            } catch (...) {
                // Silently handle failures
            }
        }
        
        // Test edge cases with special values
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Create a tensor with special values
            std::vector<int64_t> shape = {2, 2};
            torch::Tensor special_tensor;
            
            switch (edge_case % 6) {
                case 0: // Positive Infinity
                    special_tensor = torch::full(shape, std::numeric_limits<float>::infinity());
                    break;
                case 1: // Negative Infinity
                    special_tensor = torch::full(shape, -std::numeric_limits<float>::infinity());
                    break;
                case 2: // NaN
                    special_tensor = torch::full(shape, std::numeric_limits<float>::quiet_NaN());
                    break;
                case 3: // Very large positive value (cosh will overflow)
                    special_tensor = torch::full(shape, 100.0f);
                    break;
                case 4: // Very large negative value (cosh is symmetric)
                    special_tensor = torch::full(shape, -100.0f);
                    break;
                case 5: // Zero (cosh(0) = 1)
                    special_tensor = torch::zeros(shape);
                    break;
            }
            
            // Apply cosh to the special tensor
            torch::Tensor special_result = torch::cosh(special_tensor);
        }
        
        // Test with complex tensor if supported
        if (offset < Size && Data[offset] % 4 == 0) {
            try {
                torch::Tensor complex_input = input.to(torch::kComplexFloat);
                torch::Tensor complex_result = torch::cosh(complex_input);
            } catch (...) {
                // Complex may not be supported for all cases
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