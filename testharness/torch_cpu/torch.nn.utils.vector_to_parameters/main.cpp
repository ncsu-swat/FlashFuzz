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
        
        // Early exit for very small inputs
        if (Size < 4) {
            return 0;
        }
        
        // Create a vector tensor (1D tensor)
        torch::Tensor vector_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 1D tensor for vector_to_parameters
        if (vector_tensor.dim() != 1) {
            vector_tensor = vector_tensor.reshape({-1});
        }
        
        // Create a list of parameter tensors
        std::vector<torch::Tensor> parameters;
        
        // Number of parameter tensors to create (1-5)
        uint8_t num_params = 1;
        if (offset < Size) {
            num_params = (Data[offset++] % 5) + 1;
        }
        
        // Create parameter tensors with different shapes
        for (uint8_t i = 0; i < num_params && offset < Size; ++i) {
            torch::Tensor param = fuzzer_utils::createTensor(Data, Size, offset);
            parameters.push_back(param);
        }
        
        // Apply vector_to_parameters operation
        torch::nn::utils::vector_to_parameters(vector_tensor, parameters);
        
        // Test edge cases with empty parameters list
        if (offset < Size && Data[offset++] % 2 == 0) {
            std::vector<torch::Tensor> empty_params;
            torch::nn::utils::vector_to_parameters(vector_tensor, empty_params);
        }
        
        // Test with zero-sized vector
        if (offset < Size && Data[offset++] % 2 == 0) {
            torch::Tensor zero_vector = torch::zeros({0}, vector_tensor.options());
            torch::nn::utils::vector_to_parameters(zero_vector, parameters);
        }
        
        // Test with different dtypes
        if (offset < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            torch::Tensor typed_vector = vector_tensor.to(dtype);
            torch::nn::utils::vector_to_parameters(typed_vector, parameters);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
