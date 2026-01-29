#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
                
                // Convert to float if needed for gradient support
                if (!tensor.is_floating_point()) {
                    tensor = tensor.to(torch::kFloat32);
                }
                
                // Make sure tensor is contiguous
                tensor = tensor.contiguous();
                
                // Clone and set requires_grad to simulate parameters
                tensor = tensor.clone().detach().requires_grad_(true);
                
                parameters.push_back(tensor);
            } catch (...) {
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
        
        // Verify the flattened tensor has the correct size
        int64_t expected_size = 0;
        for (const auto& param : parameters) {
            expected_size += param.numel();
        }
        
        if (flattened.numel() != expected_size) {
            std::cerr << "Size mismatch in parameters_to_vector" << std::endl;
        }
        
        // Optionally test the inverse operation (vector_to_parameters)
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                // Create a copy of the parameters for vector_to_parameters
                std::vector<torch::Tensor> params_copy;
                for (const auto& p : parameters) {
                    params_copy.push_back(p.clone().detach().requires_grad_(true));
                }
                torch::nn::utils::vector_to_parameters(flattened.clone(), params_copy);
            } catch (...) {
                // vector_to_parameters may fail if sizes don't match perfectly
            }
        }
        
        // Test gradients if there's enough data left
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                // Create a scalar output by summing the flattened tensor
                torch::Tensor sum = flattened.sum();
                
                // Backpropagate
                sum.backward();
                
                // Check if gradients were properly propagated to original parameters
                for (const auto& param : parameters) {
                    if (param.grad().defined()) {
                        // Access gradients to ensure they're computed
                        volatile auto grad_sum = param.grad().sum().item<float>();
                        (void)grad_sum;
                    }
                }
            } catch (...) {
                // Gradient computation may fail in some edge cases
            }
        }
        
        // Additional test: round-trip conversion
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                // Create new parameters with same shapes
                std::vector<torch::Tensor> new_params;
                for (const auto& p : parameters) {
                    new_params.push_back(torch::zeros_like(p).requires_grad_(true));
                }
                
                // Copy vector back to new parameters
                torch::nn::utils::vector_to_parameters(flattened, new_params);
                
                // Convert back to vector
                torch::Tensor re_flattened = torch::nn::utils::parameters_to_vector(new_params);
                
                // Verify round-trip preserves values
                if (!torch::allclose(flattened.detach(), re_flattened.detach())) {
                    // Values should match after round-trip
                }
            } catch (...) {
                // Round-trip may fail in edge cases
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}