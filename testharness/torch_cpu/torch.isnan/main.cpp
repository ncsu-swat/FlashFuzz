#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        if (result.defined()) {
            bool has_nan = result.any().item<bool>();
            
            // Try some additional operations with the result
            torch::Tensor count = result.sum();
            
            // Try to convert back to original type if possible
            if (input_tensor.dtype() != torch::kBool) {
                torch::Tensor masked = input_tensor.masked_fill(result, 0);
            }
        }
        
        // If we have more data, try with a different tensor
        if (offset + 2 < Size) {
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply isnan to the second tensor
            torch::Tensor second_result = torch::isnan(second_tensor);
            
            // Try logical operations between results
            if (result.defined() && second_result.defined() && 
                result.sizes() == second_result.sizes()) {
                torch::Tensor combined = result | second_result;
            }
        }
        
        // Try with out parameter
        if (input_tensor.defined()) {
            torch::Tensor out_tensor = torch::empty_like(input_tensor, torch::kBool);
            torch::isnan_out(out_tensor, input_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}