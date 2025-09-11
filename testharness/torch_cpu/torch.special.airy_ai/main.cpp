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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.airy_ai operation
        torch::Tensor result = torch::special::airy_ai(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto item = result.item();
        }
        
        // Try with out parameter variant
        if (offset + 2 < Size) {
            torch::Tensor output = fuzzer_utils::createTensor(Data, Size, offset);
            torch::special::airy_ai_out(output, input);
            
            if (output.defined() && output.numel() > 0) {
                auto item = output.item();
            }
        }
        
        // Try with different input types
        if (offset + 2 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to different dtypes to test more edge cases
            if (input2.defined()) {
                // Try with float
                torch::Tensor float_input = input2.to(torch::kFloat);
                torch::Tensor float_result = torch::special::airy_ai(float_input);
                
                // Try with double
                torch::Tensor double_input = input2.to(torch::kDouble);
                torch::Tensor double_result = torch::special::airy_ai(double_input);
                
                // Try with complex
                if (offset + 1 < Size) {
                    try {
                        torch::Tensor complex_input = input2.to(torch::kComplexFloat);
                        torch::Tensor complex_result = torch::special::airy_ai(complex_input);
                    } catch (...) {
                        // Complex input might not be supported, ignore
                    }
                }
            }
        }
        
        // Test with extreme values
        if (offset + 2 < Size) {
            // Create tensors with extreme values
            torch::Tensor extreme_values;
            
            // Test with very large values
            extreme_values = torch::ones({2, 2}, torch::kDouble) * 1e38;
            torch::Tensor large_result = torch::special::airy_ai(extreme_values);
            
            // Test with very small values
            extreme_values = torch::ones({2, 2}, torch::kDouble) * -1e38;
            torch::Tensor small_result = torch::special::airy_ai(extreme_values);
            
            // Test with NaN and Inf
            extreme_values = torch::tensor({INFINITY, -INFINITY, NAN, 0.0}, torch::kDouble);
            torch::Tensor special_result = torch::special::airy_ai(extreme_values);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
