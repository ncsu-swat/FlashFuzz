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
        
        // Create input tensor for sinc operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply sinc operation to the input tensor
        torch::Tensor result = torch::sinc(input);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            // Create a clone to test in-place operation
            torch::Tensor input_clone = input.clone();
            
            // Test sinc operation on the clone
            torch::Tensor result2 = torch::sinc(input_clone);
            
            // Test with different input dtype if possible
            if (offset + 2 < Size) {
                uint8_t dtype_selector = Data[offset++];
                auto output_dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                try {
                    torch::Tensor converted_input = input.to(output_dtype);
                    torch::Tensor result3 = torch::sinc(converted_input);
                } catch (const std::exception&) {
                    // Ignore exceptions from dtype conversion
                }
            }
        }
        
        // Test edge cases with special values if we have enough data
        if (offset + 4 < Size) {
            // Create a small tensor with special values
            std::vector<int64_t> special_shape = {2, 2};
            torch::Tensor special_input;
            
            try {
                // Try to create a tensor with special values like inf, -inf, NaN
                special_input = torch::tensor({{0.0, INFINITY}, {-INFINITY, NAN}}, 
                                             torch::TensorOptions().dtype(torch::kFloat));
                torch::Tensor special_result = torch::sinc(special_input);
            } catch (const std::exception&) {
                // Ignore exceptions from special values
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
