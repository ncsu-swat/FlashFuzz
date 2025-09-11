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
        
        // Create a weight tensor
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for spectral_norm
        uint8_t n_power_iterations = 1;
        double eps = 1e-12;
        bool dim_is_none = true;
        int64_t dim = 0;
        
        if (offset + 3 < Size) {
            n_power_iterations = Data[offset++] % 10 + 1; // 1-10 iterations
            
            // Extract eps value (small positive number)
            uint8_t eps_exp = Data[offset++] % 20;
            eps = std::pow(10, -static_cast<double>(eps_exp));
            
            // Extract dim parameter
            dim_is_none = (Data[offset++] % 2 == 0);
            if (!dim_is_none && weight.dim() > 0) {
                dim = Data[offset++] % weight.dim();
            }
        }
        
        // Create a module to apply spectral_norm
        torch::nn::Linear linear(weight.size(0), weight.size(0));
        
        // Replace the weight with our fuzzed tensor
        linear->weight = weight;
        
        // Apply spectral_norm using functional interface
        if (dim_is_none) {
            torch::nn::functional::spectral_norm(linear->weight, n_power_iterations, eps);
        } else {
            torch::nn::functional::spectral_norm(linear->weight, n_power_iterations, eps, dim);
        }
        
        // Forward pass to trigger the spectral norm computation
        if (weight.dim() >= 2) {
            try {
                torch::Tensor input = torch::ones({1, weight.size(0)});
                torch::Tensor output = linear->forward(input);
            } catch (const std::exception &) {
                // Ignore exceptions from forward pass
            }
        }
        
        // Test spectral normalization directly on tensor
        if (weight.dim() >= 2) {
            try {
                torch::Tensor normalized = torch::nn::functional::spectral_norm(weight, n_power_iterations, eps);
            } catch (const std::exception &) {
                // Ignore exceptions from spectral norm
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
