#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create GELU module with different approximation types
        uint8_t approx_type_byte = (offset < Size) ? Data[offset++] : 0;
        std::string approximation = "none";
        
        // Select approximation type based on input data
        if (approx_type_byte % 3 == 1) {
            approximation = "tanh";
        } else if (approx_type_byte % 3 == 2) {
            approximation = "none";
        }
        
        // Create GELU module
        torch::nn::GELUOptions options;
        options.approximate(approximation);
        torch::nn::GELU gelu_module(options);
        
        // Apply GELU operation
        torch::Tensor output = gelu_module->forward(input);
        
        // Try the functional version as well
        torch::Tensor output_functional = torch::gelu(input, approximation);
        
        // Try with different input types
        if (offset + 1 < Size) {
            // Create another tensor with potentially different properties
            size_t local_offset = 0;
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, local_offset);
            
            // Try to convert to different dtype and apply GELU
            if (input2.numel() > 0) {
                uint8_t dtype_selector = Data[offset % Size];
                torch::ScalarType target_dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                try {
                    torch::Tensor converted_input = input2.to(target_dtype);
                    // GELU only supports floating point types
                    if (converted_input.is_floating_point()) {
                        torch::Tensor output2 = gelu_module->forward(converted_input);
                    }
                } catch (const std::exception&) {
                    // Conversion or operation might fail for some dtypes, that's expected
                }
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat);
            torch::Tensor empty_output = gelu_module->forward(empty_tensor);
        } catch (const std::exception&) {
            // Operation on empty tensor might fail, that's expected
        }
        
        // Try with scalar tensor
        try {
            torch::Tensor scalar_tensor = torch::tensor(3.14f);
            torch::Tensor scalar_output = gelu_module->forward(scalar_tensor);
        } catch (const std::exception&) {
            // Operation on scalar tensor might fail, that's expected
        }
        
        // Try with multidimensional tensor
        try {
            torch::Tensor multi_dim = torch::randn({2, 3, 4});
            torch::Tensor multi_output = gelu_module->forward(multi_dim);
        } catch (const std::exception&) {
            // Might fail, that's expected
        }
        
        // Try with different GELU module (opposite approximation)
        try {
            std::string other_approx = (approximation == "tanh") ? "none" : "tanh";
            torch::nn::GELUOptions other_options;
            other_options.approximate(other_approx);
            torch::nn::GELU other_gelu(other_options);
            torch::Tensor other_output = other_gelu->forward(input);
        } catch (const std::exception&) {
            // Might fail, that's expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}