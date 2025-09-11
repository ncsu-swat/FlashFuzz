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
        
        // Apply isneginf operation
        torch::Tensor result = torch::isneginf(input);
        
        // Try different variants of the API
        if (offset + 1 < Size) {
            // Create a boolean mask tensor with the same shape as input
            torch::Tensor out = torch::empty_like(input, torch::kBool);
            torch::isneginf_out(out, input);
            
            // Test the in-place version if available
            if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                // Create a copy to avoid modifying the original tensor
                torch::Tensor input_copy = input.clone();
                
                // Create a tensor with -inf values
                torch::Tensor neg_inf_mask = torch::zeros_like(input_copy);
                neg_inf_mask.index_put_({torch::randint(0, input_copy.numel(), {1})}, 
                                       torch::scalar_tensor(-std::numeric_limits<double>::infinity()));
                
                // Mix in some -inf values
                input_copy = torch::where(neg_inf_mask > 0, 
                                         torch::scalar_tensor(-std::numeric_limits<double>::infinity()),
                                         input_copy);
                
                // Apply isneginf
                torch::Tensor result2 = torch::isneginf(input_copy);
            }
        }
        
        // Test with special values if we have a floating point tensor
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
            // Create a tensor with special values
            std::vector<double> special_values = {
                -std::numeric_limits<double>::infinity(),  // -inf
                std::numeric_limits<double>::infinity(),   // +inf
                std::numeric_limits<double>::quiet_NaN(),  // NaN
                -0.0,                                      // -0
                0.0,                                       // +0
                -1.0,                                      // negative number
                1.0                                        // positive number
            };
            
            auto options = torch::TensorOptions().dtype(input.dtype());
            torch::Tensor special_tensor = torch::tensor(special_values, options);
            
            // Apply isneginf to special values
            torch::Tensor special_result = torch::isneginf(special_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
