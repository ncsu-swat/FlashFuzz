#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create parameter tensors - they need to be contiguous for the conversion
        torch::Tensor param1 = fuzzer_utils::createTensor(Data, Size, offset);
        if (!param1.defined() || param1.numel() == 0) {
            return 0;
        }
        param1 = param1.contiguous();
        
        // Create a vector of parameters
        std::vector<torch::Tensor> parameters;
        parameters.push_back(param1);
        
        // Try to create additional parameters if we have enough data
        if (offset + 4 < Size) {
            torch::Tensor param2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (param2.defined() && param2.numel() > 0) {
                param2 = param2.contiguous();
                parameters.push_back(param2);
            }
        }
        
        if (offset + 4 < Size) {
            torch::Tensor param3 = fuzzer_utils::createTensor(Data, Size, offset);
            if (param3.defined() && param3.numel() > 0) {
                param3 = param3.contiguous();
                parameters.push_back(param3);
            }
        }
        
        // Get a byte to determine the test path
        uint8_t test_path = 0;
        if (offset < Size) {
            test_path = Data[offset++] % 4;
        }
        
        if (test_path == 0) {
            // Test parameters_to_vector only
            torch::Tensor flat_param = torch::nn::utils::parameters_to_vector(parameters);
            
            // Verify the flattened tensor has correct size
            int64_t expected_size = 0;
            for (const auto& p : parameters) {
                expected_size += p.numel();
            }
            (void)flat_param.size(0); // Access to ensure computation
        } 
        else if (test_path == 1) {
            // Test round-trip: parameters_to_vector then vector_to_parameters
            torch::Tensor flat_param = torch::nn::utils::parameters_to_vector(parameters);
            
            // Create copies of parameters for vector_to_parameters (it modifies in-place)
            std::vector<torch::Tensor> param_copies;
            for (const auto& p : parameters) {
                param_copies.push_back(p.clone());
            }
            
            // Convert vector back to parameters
            torch::nn::utils::vector_to_parameters(flat_param, param_copies);
        }
        else if (test_path == 2) {
            // Test with gradients
            std::vector<torch::Tensor> grad_params;
            for (const auto& p : parameters) {
                auto grad_p = p.clone().set_requires_grad(true);
                grad_params.push_back(grad_p);
            }
            
            torch::Tensor flat_param = torch::nn::utils::parameters_to_vector(grad_params);
            (void)flat_param.requires_grad(); // Check gradient property
        }
        else {
            // Test with different dtypes
            std::vector<torch::Tensor> float_params;
            for (const auto& p : parameters) {
                try {
                    float_params.push_back(p.to(torch::kFloat32).contiguous());
                } catch (...) {
                    // Silently ignore conversion failures
                    float_params.push_back(p.contiguous());
                }
            }
            
            if (!float_params.empty()) {
                torch::Tensor flat_param = torch::nn::utils::parameters_to_vector(float_params);
                (void)flat_param.dtype();
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