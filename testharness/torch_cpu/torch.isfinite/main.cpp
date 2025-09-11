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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply isfinite operation
        torch::Tensor result = torch::isfinite(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            bool has_true = result.any().item<bool>();
            bool has_false = torch::logical_not(result).any().item<bool>();
            
            // Try some additional operations with the result
            torch::Tensor sum_result = result.sum();
            torch::Tensor mean_result = result.to(torch::kFloat).mean();
        }
        
        // If we have more data, try creating another tensor and test with it
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a tensor with special values (inf, -inf, nan)
            auto options = torch::TensorOptions().dtype(another_tensor.dtype());
            if (another_tensor.dtype() == torch::kFloat || 
                another_tensor.dtype() == torch::kDouble ||
                another_tensor.dtype() == torch::kHalf ||
                another_tensor.dtype() == torch::kBFloat16) {
                
                // Create a tensor with the same shape but containing special values
                torch::Tensor special_values;
                if (another_tensor.numel() > 0) {
                    special_values = torch::empty_like(another_tensor);
                    
                    // Fill with a mix of finite, infinite, and NaN values
                    auto accessor = special_values.accessor<float, 1>();
                    for (int64_t i = 0; i < special_values.numel(); i++) {
                        int val = (i % 4);
                        if (val == 0) accessor[i] = 1.0;       // finite
                        else if (val == 1) accessor[i] = INFINITY;  // inf
                        else if (val == 2) accessor[i] = -INFINITY; // -inf
                        else accessor[i] = NAN;                // nan
                    }
                    
                    // Apply isfinite to the special values tensor
                    torch::Tensor special_result = torch::isfinite(special_values);
                }
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
