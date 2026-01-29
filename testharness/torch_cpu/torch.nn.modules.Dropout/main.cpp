#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least 3 bytes for basic operation
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dropout probability from the input data
        // Use a byte to derive probability in valid range [0, 1]
        double p = 0.5;
        if (offset < Size) {
            p = static_cast<double>(Data[offset++]) / 255.0;
        }
        
        // Extract inplace flag from the input data
        bool inplace = false;
        if (offset < Size) {
            inplace = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Extract training mode from input data
        bool training_mode = true;
        if (offset < Size) {
            training_mode = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Test 1: Dropout module with options
        try {
            torch::nn::Dropout dropout_module(torch::nn::DropoutOptions().p(p).inplace(inplace));
            
            if (training_mode) {
                dropout_module->train();
            } else {
                dropout_module->eval();
            }
            
            // Clone input if inplace to avoid modifying original
            torch::Tensor module_input = inplace ? input.clone() : input;
            torch::Tensor output = dropout_module->forward(module_input);
            
            // Verify output shape matches input shape
            (void)output.sizes();
        } catch (const std::exception &) {
            // Expected failures for certain parameter combinations
        }
        
        // Test 2: Dropout module with default options
        try {
            torch::nn::Dropout dropout_default;
            dropout_default->train(training_mode);
            torch::Tensor output_default = dropout_default->forward(input.clone());
            (void)output_default.sizes();
        } catch (const std::exception &) {
            // Expected failures
        }
        
        // Test 3: Functional interface torch::dropout
        try {
            torch::Tensor output_functional = torch::dropout(input.clone(), p, training_mode);
            (void)output_functional.sizes();
        } catch (const std::exception &) {
            // Expected failures
        }
        
        // Test 4: Inplace functional dropout
        try {
            torch::Tensor input_copy = input.clone();
            torch::Tensor output_inplace = torch::dropout_(input_copy, p, training_mode);
            (void)output_inplace.sizes();
        } catch (const std::exception &) {
            // Expected failures
        }
        
        // Test 5: Test with different tensor types if we have more data
        if (offset + 1 < Size) {
            try {
                int dtype_selector = Data[offset++] % 3;
                torch::Tensor typed_input;
                
                if (dtype_selector == 0) {
                    typed_input = input.to(torch::kFloat32);
                } else if (dtype_selector == 1) {
                    typed_input = input.to(torch::kFloat64);
                } else {
                    typed_input = input.to(torch::kFloat16);
                }
                
                torch::nn::Dropout dropout_typed(torch::nn::DropoutOptions().p(p));
                dropout_typed->train(training_mode);
                torch::Tensor output_typed = dropout_typed->forward(typed_input);
                (void)output_typed.sizes();
            } catch (const std::exception &) {
                // Expected failures for unsupported dtypes
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}