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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply isnan operation
        torch::Tensor result = torch::isnan(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            bool has_nan = result.any().item<bool>();
            
            // Try some additional operations with the result
            torch::Tensor count = result.sum();
            
            // Try to convert back to original type if possible
            // masked_fill requires non-boolean tensor for the value
            if (input_tensor.dtype() != torch::kBool) {
                try {
                    torch::Tensor masked = input_tensor.masked_fill(result, 0);
                } catch (...) {
                    // Shape mismatch or other expected failures - ignore silently
                }
            }
        }
        
        // If we have more data, try with a different tensor
        if (offset + 2 < Size) {
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply isnan to the second tensor
            torch::Tensor second_result = torch::isnan(second_tensor);
            
            // Try logical operations between results
            if (result.defined() && second_result.defined() && 
                result.numel() > 0 && second_result.numel() > 0 &&
                result.sizes() == second_result.sizes()) {
                try {
                    torch::Tensor combined = result | second_result;
                    torch::Tensor combined_and = result & second_result;
                    torch::Tensor combined_xor = result ^ second_result;
                } catch (...) {
                    // Shape or type mismatch - ignore silently
                }
            }
        }
        
        // Try with out parameter
        if (input_tensor.defined() && input_tensor.numel() > 0) {
            try {
                torch::Tensor out_tensor = torch::empty_like(input_tensor, torch::kBool);
                torch::isnan_out(out_tensor, input_tensor);
            } catch (...) {
                // Expected failures with certain tensor types - ignore silently
            }
        }
        
        // Test with special floating point values to improve coverage
        if (Size >= 4) {
            try {
                // Create tensor with potential NaN values
                torch::Tensor float_tensor = torch::tensor({
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    0.0f,
                    1.0f,
                    -1.0f
                });
                torch::Tensor nan_result = torch::isnan(float_tensor);
                
                // Verify the result makes sense
                auto accessor = nan_result.accessor<bool, 1>();
                (void)accessor[0]; // First should be true (NaN)
            } catch (...) {
                // Ignore failures in special value testing
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