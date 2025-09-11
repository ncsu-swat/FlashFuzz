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
        
        // Create input tensor
        if (offset < Size) {
            torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Parse replacement values for nan, posinf, neginf
            double nan_value = 0.0;
            double posinf_value = 0.0;
            double neginf_value = 0.0;
            
            // Parse nan replacement if we have enough data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&nan_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Parse posinf replacement if we have enough data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&posinf_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Parse neginf replacement if we have enough data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&neginf_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Make a copy of the input tensor to verify the in-place operation
            torch::Tensor input_copy = input_tensor.clone();
            
            // Apply nan_to_num_ in-place operation
            input_tensor.nan_to_num_(nan_value, posinf_value, neginf_value);
            
            // Verify the operation by comparing with non-in-place version
            torch::Tensor expected = input_copy.nan_to_num(nan_value, posinf_value, neginf_value);
            
            // Check if the in-place operation produced the expected result
            if (!torch::allclose(input_tensor, expected, 1e-5, 1e-8)) {
                throw std::runtime_error("nan_to_num_ produced unexpected result");
            }
            
            // Test with default parameters
            torch::Tensor default_test = input_copy.clone();
            default_test.nan_to_num_();
            
            // Test with partial parameters
            torch::Tensor partial_test = input_copy.clone();
            partial_test.nan_to_num_(0.0);
            
            // Test with different tensor types if possible
            if (input_tensor.dtype() != torch::kFloat) {
                torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
                float_tensor.nan_to_num_();
            }
            
            if (input_tensor.dtype() != torch::kDouble) {
                torch::Tensor double_tensor = input_tensor.to(torch::kDouble);
                double_tensor.nan_to_num_();
            }
            
            // Test with empty tensor
            torch::Tensor empty_tensor = torch::empty({0}, input_tensor.options());
            empty_tensor.nan_to_num_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
