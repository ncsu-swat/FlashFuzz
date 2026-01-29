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
        
        // Apply asinh operation
        torch::Tensor result = torch::asinh(input);
        
        // Try in-place version if there's more data
        if (offset < Size && Data[offset] % 2 == 0) {
            // In-place requires float type
            torch::Tensor input_copy = input.to(torch::kFloat32).clone();
            input_copy.asinh_();
        }
        
        // Try with different output types if there's more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                // Apply asinh and convert to desired dtype (may fail for some types)
                torch::Tensor result_with_dtype = torch::asinh(input).to(dtype);
            } catch (...) {
                // Silently ignore type conversion failures
            }
        }
        
        // Try with out parameter if there's more data
        if (offset < Size && Data[offset] % 5 == 0) {
            try {
                torch::Tensor out = torch::empty_like(input);
                torch::asinh_out(out, input);
            } catch (...) {
                // Silently ignore out parameter failures (dtype mismatch, etc.)
            }
        }
        
        // Test with specific tensor types to improve coverage
        if (offset < Size) {
            uint8_t test_selector = Data[offset++];
            
            try {
                if (test_selector % 4 == 0) {
                    // Test with float tensor
                    auto float_input = torch::randn({3, 3}, torch::kFloat32);
                    torch::asinh(float_input);
                } else if (test_selector % 4 == 1) {
                    // Test with double tensor
                    auto double_input = torch::randn({2, 4}, torch::kFloat64);
                    torch::asinh(double_input);
                } else if (test_selector % 4 == 2) {
                    // Test with complex tensor
                    auto complex_input = torch::randn({2, 2}, torch::kComplexFloat);
                    torch::asinh(complex_input);
                } else {
                    // Test with 1D tensor
                    auto vec_input = torch::randn({10}, torch::kFloat32);
                    torch::asinh(vec_input);
                }
            } catch (...) {
                // Silently ignore failures from fixed test cases
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