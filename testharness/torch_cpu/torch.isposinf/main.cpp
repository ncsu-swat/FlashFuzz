#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply isposinf operation
        torch::Tensor result = torch::isposinf(input);
        
        // Try different variants of the API
        if (offset + 1 < Size) {
            // Create a boolean tensor to store the result
            torch::Tensor out = torch::empty_like(input, torch::kBool);
            torch::isposinf_out(out, input);
            
            // Test the in-place version if available
            if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                torch::Tensor clone = input.clone();
                torch::Tensor result2 = torch::isposinf(clone);
            }
        }
        
        // Test edge cases with special values if we have float or double tensors
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
            // Create a tensor with special values
            std::vector<float> special_values = {
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f,
                -0.0f,
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::min(),
                std::numeric_limits<float>::lowest()
            };
            
            torch::Tensor special_tensor = torch::tensor(special_values);
            torch::Tensor special_result = torch::isposinf(special_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}