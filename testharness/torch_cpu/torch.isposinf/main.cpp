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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply isposinf operation - returns boolean tensor indicating positive infinity
        torch::Tensor result = torch::isposinf(input);
        
        // Verify result is boolean type
        if (result.dtype() != torch::kBool) {
            std::cerr << "Unexpected result dtype" << std::endl;
        }
        
        // Test with explicit output tensor using torch::isposinf with out parameter
        try {
            torch::Tensor out = torch::empty_like(input, torch::kBool);
            torch::isposinf_out(out, input);
        } catch (...) {
            // isposinf_out may not exist or may fail for certain dtypes, ignore
        }
        
        // Test with different tensor types based on fuzzer data
        if (offset < Size) {
            uint8_t type_selector = Data[offset++] % 4;
            torch::Tensor typed_input;
            
            try {
                switch (type_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat);
                        break;
                    case 1:
                        typed_input = input.to(torch::kDouble);
                        break;
                    case 2:
                        typed_input = input.to(torch::kHalf);
                        break;
                    default:
                        typed_input = input.to(torch::kBFloat16);
                        break;
                }
                torch::Tensor typed_result = torch::isposinf(typed_input);
            } catch (...) {
                // Type conversion may fail for some inputs, ignore
            }
        }
        
        // Test edge cases with special float values
        if (offset < Size && (Data[offset] % 2 == 0)) {
            std::vector<float> special_values = {
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f,
                -0.0f,
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::min(),
                std::numeric_limits<float>::lowest(),
                1.0f,
                -1.0f
            };
            
            torch::Tensor special_tensor = torch::tensor(special_values);
            torch::Tensor special_result = torch::isposinf(special_tensor);
            
            // Also test double precision
            torch::Tensor special_double = special_tensor.to(torch::kDouble);
            torch::Tensor special_double_result = torch::isposinf(special_double);
        }
        
        // Test with multi-dimensional tensors
        if (offset + 2 < Size) {
            int dim1 = (Data[offset] % 5) + 1;
            int dim2 = (Data[offset + 1] % 5) + 1;
            offset += 2;
            
            try {
                torch::Tensor reshaped = input.reshape({-1}).slice(0, 0, std::min((int64_t)(dim1 * dim2), input.numel()));
                if (reshaped.numel() == dim1 * dim2) {
                    torch::Tensor multi_dim = reshaped.reshape({dim1, dim2});
                    torch::Tensor multi_result = torch::isposinf(multi_dim);
                }
            } catch (...) {
                // Reshape may fail, ignore
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