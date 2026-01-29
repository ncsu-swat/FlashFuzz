#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor with requires_grad for gradient operations
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        input_tensor = input_tensor.clone().detach().requires_grad_(true);
        
        // Create a parameter tensor
        torch::Tensor param_tensor;
        if (offset < Size - 4) {
            param_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            param_tensor = torch::randn({4, 4});
        }
        param_tensor = param_tensor.clone().detach().requires_grad_(true);
        
        // Get a byte to determine which nn.utils function to test
        uint8_t function_selector = 0;
        if (offset < Size) {
            function_selector = Data[offset++];
        }
        
        // Test various torch.nn.utils functions
        switch (function_selector % 4) {
            case 0: {
                // Test clip_grad_norm_
                // First, we need to create gradients by doing a backward pass
                torch::Tensor loss = input_tensor.sum() + param_tensor.sum();
                loss.backward();
                
                std::vector<torch::Tensor> parameters = {input_tensor, param_tensor};
                
                // Parse max_norm from fuzzer data, ensure it's positive and finite
                double max_norm = 1.0;
                if (offset + sizeof(float) <= Size) {
                    float raw_val;
                    std::memcpy(&raw_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    if (std::isfinite(raw_val) && raw_val > 0) {
                        max_norm = static_cast<double>(raw_val);
                    }
                }
                
                // Parse norm_type, constrain to reasonable values
                double norm_type = 2.0;
                if (offset < Size) {
                    norm_type = static_cast<double>((Data[offset++] % 4) + 1); // 1, 2, 3, or 4
                }
                
                torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type);
                break;
            }
            
            case 1: {
                // Test clip_grad_value_
                // Create gradients first
                torch::Tensor loss = input_tensor.sum() + param_tensor.sum();
                loss.backward();
                
                std::vector<torch::Tensor> parameters = {input_tensor, param_tensor};
                
                // Parse clip_value, ensure it's positive and finite
                double clip_value = 1.0;
                if (offset + sizeof(float) <= Size) {
                    float raw_val;
                    std::memcpy(&raw_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    if (std::isfinite(raw_val) && raw_val > 0) {
                        clip_value = static_cast<double>(raw_val);
                    }
                }
                
                torch::nn::utils::clip_grad_value_(parameters, clip_value);
                break;
            }
            
            case 2: {
                // Test parameters_to_vector
                std::vector<torch::Tensor> parameters = {input_tensor, param_tensor};
                torch::Tensor vec = torch::nn::utils::parameters_to_vector(parameters);
                
                // Verify result
                int64_t expected_size = input_tensor.numel() + param_tensor.numel();
                (void)expected_size; // Suppress unused warning
                break;
            }
            
            case 3: {
                // Test vector_to_parameters
                // Create a vector that matches the total size of parameters
                int64_t total_numel = param_tensor.numel();
                torch::Tensor vec = torch::randn({total_numel});
                
                // Use fuzzer data to populate vector if available
                if (offset + 4 <= Size) {
                    try {
                        torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        vec = fuzz_tensor.flatten().narrow(0, 0, std::min(fuzz_tensor.numel(), total_numel));
                        if (vec.numel() < total_numel) {
                            // Pad if necessary
                            vec = torch::cat({vec, torch::zeros({total_numel - vec.numel()})});
                        }
                    } catch (...) {
                        // Use default random vector on failure
                    }
                }
                
                std::vector<torch::Tensor> parameters = {param_tensor};
                torch::nn::utils::vector_to_parameters(vec, parameters);
                break;
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