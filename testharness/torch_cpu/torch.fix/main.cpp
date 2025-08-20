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
        
        // Apply torch.fix operation
        torch::Tensor result = torch::fix(input);
        
        // Try different variants of the operation
        if (offset < Size) {
            // Try in-place version if we have more data
            torch::Tensor input_copy = input.clone();
            input_copy.fix_();
        }
        
        // Try with out parameter if we have more data
        if (offset < Size) {
            torch::Tensor output = torch::empty_like(input);
            torch::fix_out(output, input);
        }
        
        // Try with named parameters if we have more data
        if (offset < Size) {
            torch::Tensor output = torch::empty_like(input);
            torch::fix_out(output, input);
        }
        
        // Try with different tensor types if we have more data
        if (offset < Size) {
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::Tensor result_float = torch::fix(float_input);
        }
        
        // Try with complex numbers if we have more data
        if (offset < Size && input.is_complex()) {
            torch::Tensor complex_result = torch::fix(input);
        }
        
        // Try with boolean tensors if we have more data
        if (offset < Size && input.dtype() == torch::kBool) {
            torch::Tensor bool_result = torch::fix(input);
        }
        
        // Try with empty tensors if we have more data
        if (offset < Size) {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_result = torch::fix(empty_tensor);
        }
        
        // Try with scalar tensors if we have more data
        if (offset < Size) {
            torch::Tensor scalar_tensor = torch::tensor(3.7);
            torch::Tensor scalar_result = torch::fix(scalar_tensor);
        }
        
        // Try with negative values if we have more data
        if (offset < Size) {
            torch::Tensor neg_tensor = torch::tensor({-3.7, -2.1, -0.9, 0.0, 0.9, 2.1, 3.7});
            torch::Tensor neg_result = torch::fix(neg_tensor);
        }
        
        // Try with NaN and Inf values if we have more data
        if (offset < Size) {
            torch::Tensor special_tensor = torch::tensor({std::numeric_limits<float>::quiet_NaN(), 
                                                         std::numeric_limits<float>::infinity(),
                                                         -std::numeric_limits<float>::infinity()});
            torch::Tensor special_result = torch::fix(special_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}