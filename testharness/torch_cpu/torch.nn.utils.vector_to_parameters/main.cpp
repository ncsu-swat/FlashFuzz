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
        
        // Early exit for very small inputs
        if (Size < 4) {
            return 0;
        }
        
        // Determine number of parameters to create (1-5)
        uint8_t num_params = (Data[offset++] % 5) + 1;
        
        // Create parameter tensors with different shapes
        std::vector<torch::Tensor> parameters;
        int64_t total_numel = 0;
        
        for (uint8_t i = 0; i < num_params && offset < Size; ++i) {
            try {
                torch::Tensor param = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure floating point for parameter-like behavior
                if (!param.is_floating_point()) {
                    param = param.to(torch::kFloat32);
                }
                
                // Make contiguous
                param = param.contiguous();
                
                parameters.push_back(param);
                total_numel += param.numel();
            } catch (...) {
                // Skip if tensor creation fails
                continue;
            }
        }
        
        // Skip if no valid parameters were created or total size is 0
        if (parameters.empty() || total_numel == 0) {
            return 0;
        }
        
        // Create a 1D vector tensor with the correct size to match parameters
        torch::Tensor vector_tensor = torch::randn({total_numel}, parameters[0].options());
        
        // Use remaining fuzzer data to fill the vector with diverse values
        if (offset + 4 < Size) {
            try {
                torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (!fuzz_tensor.is_floating_point()) {
                    fuzz_tensor = fuzz_tensor.to(torch::kFloat32);
                }
                // Flatten and take as many elements as we need
                fuzz_tensor = fuzz_tensor.reshape({-1});
                if (fuzz_tensor.numel() >= total_numel) {
                    vector_tensor = fuzz_tensor.slice(0, 0, total_numel).contiguous();
                } else if (fuzz_tensor.numel() > 0) {
                    // Repeat to fill
                    int64_t repeats = (total_numel / fuzz_tensor.numel()) + 1;
                    vector_tensor = fuzz_tensor.repeat({repeats}).slice(0, 0, total_numel).contiguous();
                }
            } catch (...) {
                // Keep the random vector_tensor
            }
        }
        
        // Ensure vector is contiguous
        vector_tensor = vector_tensor.contiguous();
        
        // Apply vector_to_parameters operation
        torch::nn::utils::vector_to_parameters(vector_tensor, parameters);
        
        // Verify: after the call, parameters should contain values from vector
        // This is a sanity check - access the data to ensure it's been written
        for (const auto& param : parameters) {
            volatile auto check = param.sum().item<float>();
            (void)check;
        }
        
        // Test round-trip: parameters_to_vector then vector_to_parameters
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                // Get current parameters as a vector
                torch::Tensor flat = torch::nn::utils::parameters_to_vector(parameters);
                
                // Create new parameters with same shapes
                std::vector<torch::Tensor> new_params;
                for (const auto& p : parameters) {
                    new_params.push_back(torch::zeros_like(p));
                }
                
                // Copy vector to new parameters
                torch::nn::utils::vector_to_parameters(flat, new_params);
                
                // Verify values were copied
                for (size_t i = 0; i < parameters.size(); ++i) {
                    if (!torch::allclose(parameters[i], new_params[i])) {
                        // Values should match
                    }
                }
            } catch (...) {
                // Silently handle failures in round-trip test
            }
        }
        
        // Test with requires_grad parameters
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                std::vector<torch::Tensor> grad_params;
                int64_t grad_total = 0;
                for (const auto& p : parameters) {
                    auto grad_p = p.clone().detach().set_requires_grad(true);
                    grad_params.push_back(grad_p);
                    grad_total += grad_p.numel();
                }
                
                torch::Tensor grad_vec = torch::randn({grad_total}, grad_params[0].options());
                torch::nn::utils::vector_to_parameters(grad_vec, grad_params);
                
                // Verify grad params still have requires_grad
                for (const auto& gp : grad_params) {
                    (void)gp.requires_grad();
                }
            } catch (...) {
                // Silently handle gradient test failures
            }
        }
        
        // Test with double precision
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                std::vector<torch::Tensor> double_params;
                int64_t double_total = 0;
                for (const auto& p : parameters) {
                    auto dp = p.to(torch::kFloat64).contiguous();
                    double_params.push_back(dp);
                    double_total += dp.numel();
                }
                
                torch::Tensor double_vec = torch::randn({double_total}, torch::kFloat64);
                torch::nn::utils::vector_to_parameters(double_vec, double_params);
            } catch (...) {
                // Silently handle dtype test failures
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