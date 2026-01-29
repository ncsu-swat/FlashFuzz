#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Apply Hardswish using functional API
        torch::Tensor output = torch::hardswish(input);
        
        // Try inplace version if there's enough data to determine whether to use it
        if (offset < Size) {
            bool use_inplace = Data[offset++] % 2 == 0;
            if (use_inplace) {
                torch::Tensor input_clone = input.clone();
                torch::hardswish_(input_clone);
            }
        }
        
        // Test with different tensor types/shapes
        if (offset < Size) {
            bool test_float_tensor = Data[offset++] % 2 == 0;
            if (test_float_tensor) {
                // Create a float tensor explicitly and apply hardswish
                torch::Tensor float_input = input.to(torch::kFloat32);
                torch::Tensor float_output = torch::hardswish(float_input);
                
                // Also test inplace on float tensor
                torch::hardswish_(float_input);
            }
        }
        
        // Try with different tensor options if there's more data
        if (offset + 1 < Size) {
            // Create a new tensor with different options
            size_t new_offset = offset;
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, new_offset);
            
            // Apply Hardswish to this new tensor
            torch::Tensor another_output = torch::hardswish(another_input);
        }
        
        // Test edge cases with specific tensor values if we have more data
        if (offset < Size) {
            int test_case = Data[offset++] % 4;
            torch::Tensor test_tensor;
            
            switch (test_case) {
                case 0:
                    // Values around the transition points (-3 and 3)
                    test_tensor = torch::tensor({-4.0f, -3.0f, -2.0f, 0.0f, 2.0f, 3.0f, 4.0f});
                    break;
                case 1:
                    // Very large values
                    test_tensor = torch::tensor({-1e6f, 1e6f});
                    break;
                case 2:
                    // Very small values
                    test_tensor = torch::tensor({-1e-6f, 1e-6f});
                    break;
                case 3:
                    // Special float values (inf, -inf handled by inner try-catch)
                    test_tensor = torch::tensor({0.0f, -0.0f});
                    break;
            }
            
            torch::Tensor test_output = torch::hardswish(test_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}