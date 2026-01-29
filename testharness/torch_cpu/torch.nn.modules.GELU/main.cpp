#include "fuzzer_utils.h"
#include <iostream>

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
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create GELU module with different approximation types
        uint8_t approx_type_byte = (offset < Size) ? Data[offset++] : 0;
        std::string approximation;
        
        // Select approximation type based on input data
        // Valid values are "none" and "tanh"
        if (approx_type_byte % 2 == 0) {
            approximation = "none";
        } else {
            approximation = "tanh";
        }
        
        // Create GELU module with options
        torch::nn::GELUOptions options;
        options.approximate(approximation);
        torch::nn::GELU gelu_module(options);
        
        // Apply GELU operation via module forward
        torch::Tensor output = gelu_module->forward(input);
        
        // Test with different tensor types to improve coverage
        try {
            // Test with float tensor
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor float_output = gelu_module->forward(float_input);
            
            // Test with double tensor
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor double_output = gelu_module->forward(double_input);
        } catch (...) {
            // Silently ignore dtype conversion issues
        }
        
        // Try the functional version as well for additional coverage
        torch::Tensor output_functional = torch::gelu(input, approximation);
        
        // Test inplace-like behavior by reusing output
        try {
            torch::Tensor chained = gelu_module->forward(output);
        } catch (...) {
            // Silently ignore
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}