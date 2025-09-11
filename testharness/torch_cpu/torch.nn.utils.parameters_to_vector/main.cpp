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
        
        // Determine number of parameters to create
        if (Size < 1) return 0;
        uint8_t num_params_byte = Data[offset++];
        int num_params = (num_params_byte % 10) + 1; // Create 1-10 parameters
        
        // Create a vector of parameters (tensors)
        std::vector<torch::Tensor> parameters;
        
        // Create tensors with different shapes and dtypes
        for (int i = 0; i < num_params && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Make the tensor require gradients to simulate parameters
                tensor.set_requires_grad(true);
                
                parameters.push_back(tensor);
            } catch (const std::exception& e) {
                // If one tensor creation fails, continue with the next
                continue;
            }
        }
        
        // Skip if no valid parameters were created
        if (parameters.empty()) {
            return 0;
        }
        
        // Apply parameters_to_vector operation
        torch::Tensor flattened = torch::nn::utils::parameters_to_vector(parameters);
        
        // Optionally test the inverse operation (vector_to_parameters)
        if (offset < Size && Data[offset++] % 2 == 0) {
            torch::nn::utils::vector_to_parameters(flattened, parameters);
        }
        
        // Test gradients if there's enough data left
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Create a scalar output by summing the flattened tensor
            torch::Tensor sum = flattened.sum();
            
            // Backpropagate
            sum.backward();
            
            // Check if gradients were properly propagated to original parameters
            for (auto& param : parameters) {
                if (param.grad().defined()) {
                    // Access gradients to ensure they're computed
                    auto grad_sum = param.grad().sum().item<float>();
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
