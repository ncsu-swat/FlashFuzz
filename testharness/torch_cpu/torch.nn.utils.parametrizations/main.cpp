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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple linear module to test parametrizations
        torch::nn::Linear linear_module(input_tensor.size(-1), 10);
        
        // Get a byte to determine which parametrization to apply
        uint8_t param_type = 0;
        if (offset < Size) {
            param_type = Data[offset++];
        }
        
        // Apply different parametrizations based on the byte
        switch (param_type % 4) {
            case 0: {
                // Spectral norm parametrization
                auto parametrized = torch::nn::utils::spectral_norm(linear_module);
                
                // Forward pass through the parametrized module
                if (input_tensor.dim() > 0 && input_tensor.size(0) > 0) {
                    auto output = parametrized->forward(input_tensor);
                }
                
                // Test removing the parametrization
                torch::nn::utils::remove_spectral_norm(parametrized);
                break;
            }
            
            case 1: {
                // Weight norm parametrization
                auto parametrized = torch::nn::utils::weight_norm(linear_module);
                
                // Forward pass through the parametrized module
                if (input_tensor.dim() > 0 && input_tensor.size(0) > 0) {
                    auto output = parametrized->forward(input_tensor);
                }
                
                // Test removing the parametrization
                torch::nn::utils::remove_weight_norm(parametrized);
                break;
            }
            
            case 2: {
                // Weight norm with dimension parameter
                int64_t dim = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                
                auto parametrized = torch::nn::utils::weight_norm(linear_module, dim);
                
                // Forward pass through the parametrized module
                if (input_tensor.dim() > 0 && input_tensor.size(0) > 0) {
                    auto output = parametrized->forward(input_tensor);
                }
                
                // Test removing the parametrization
                torch::nn::utils::remove_weight_norm(parametrized);
                break;
            }
            
            case 3: {
                // Spectral norm with name parameter
                auto parametrized = torch::nn::utils::spectral_norm(linear_module, "weight");
                
                // Forward pass through the parametrized module
                if (input_tensor.dim() > 0 && input_tensor.size(0) > 0) {
                    auto output = parametrized->forward(input_tensor);
                }
                
                // Test removing the parametrization
                torch::nn::utils::remove_spectral_norm(parametrized);
                break;
            }
        }
        
        // Test with additional parameters if we have more data
        if (offset + 8 < Size) {
            // Get n_power_iterations
            int64_t n_power_iterations = 1;
            std::memcpy(&n_power_iterations, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            n_power_iterations = std::abs(n_power_iterations) % 10 + 1;  // Keep it reasonable
            
            // Get eps
            double eps = 1e-12;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&eps, Data + offset, sizeof(double));
                offset += sizeof(double);
                // Ensure eps is positive and reasonable
                eps = std::abs(eps);
                if (eps == 0.0) eps = 1e-12;
                if (eps > 0.1) eps = 0.1;
            }
            
            // Apply spectral norm with additional parameters
            auto parametrized = torch::nn::utils::spectral_norm(
                linear_module, "weight", n_power_iterations, eps, true);
            
            // Forward pass
            if (input_tensor.dim() > 0 && input_tensor.size(0) > 0) {
                auto output = parametrized->forward(input_tensor);
            }
            
            // Remove parametrization
            torch::nn::utils::remove_spectral_norm(parametrized);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
