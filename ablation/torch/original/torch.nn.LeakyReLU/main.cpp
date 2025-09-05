#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes: 1 for negative_slope selector, 1 for inplace flag, 1 for tensor metadata
        if (Size < 3)
        {
            return 0; // Not enough data to work with
        }
        
        // Parse negative_slope from first byte
        // Map byte to a range of interesting values including edge cases
        uint8_t slope_selector = Data[offset++];
        float negative_slope;
        
        // Create diverse negative_slope values including edge cases
        switch (slope_selector % 10)
        {
            case 0:
                negative_slope = 0.01f; // Default value
                break;
            case 1:
                negative_slope = 0.0f; // Zero slope
                break;
            case 2:
                negative_slope = 1.0f; // Unity slope
                break;
            case 3:
                negative_slope = -1.0f; // Negative slope
                break;
            case 4:
                negative_slope = 0.1f; // Common value
                break;
            case 5:
                negative_slope = std::numeric_limits<float>::infinity(); // Infinity
                break;
            case 6:
                negative_slope = -std::numeric_limits<float>::infinity(); // Negative infinity
                break;
            case 7:
                negative_slope = std::numeric_limits<float>::quiet_NaN(); // NaN
                break;
            case 8:
                negative_slope = std::numeric_limits<float>::min(); // Smallest positive
                break;
            case 9:
                // Use raw bytes for arbitrary float
                if (offset + sizeof(float) <= Size)
                {
                    std::memcpy(&negative_slope, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                else
                {
                    negative_slope = 0.01f;
                }
                break;
        }
        
        // Parse inplace flag
        bool inplace = (Data[offset++] % 2) == 1;
        
        // Create input tensor from remaining data
        torch::Tensor input;
        try
        {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        }
        catch (const std::exception &e)
        {
            // If tensor creation fails, try with a simple random tensor
            input = torch::randn({2, 3});
        }
        
        // Create LeakyReLU module
        torch::nn::LeakyReLU leaky_relu_module(torch::nn::LeakyReLUOptions()
            .negative_slope(negative_slope)
            .inplace(inplace));
        
        // Test 1: Apply LeakyReLU through module
        torch::Tensor output;
        if (inplace && input.is_floating_point())
        {
            // For inplace operation, clone the input first to preserve original for comparison
            torch::Tensor input_clone = input.clone();
            output = leaky_relu_module->forward(input_clone);
            
            // Verify inplace operation worked correctly
            if (!output.is_same(input_clone))
            {
                std::cerr << "Inplace operation did not work as expected" << std::endl;
            }
        }
        else
        {
            output = leaky_relu_module->forward(input);
        }
        
        // Test 2: Also test functional API
        torch::Tensor functional_output = torch::nn::functional::leaky_relu(
            input,
            torch::nn::functional::LeakyReLUFuncOptions()
                .negative_slope(negative_slope)
                .inplace(false) // Always use non-inplace for functional test
        );
        
        // Test 3: Test with different tensor types and edge cases
        if (input.numel() > 0)
        {
            // Test with zeros
            torch::Tensor zeros = torch::zeros_like(input);
            torch::Tensor zeros_output = leaky_relu_module->forward(zeros);
            
            // Test with positive values
            torch::Tensor positive = torch::abs(input) + 1e-6;
            torch::Tensor positive_output = leaky_relu_module->forward(positive);
            
            // Test with negative values
            torch::Tensor negative = -torch::abs(input) - 1e-6;
            torch::Tensor negative_output = leaky_relu_module->forward(negative);
            
            // Test with mixed values
            if (input.numel() > 1)
            {
                torch::Tensor mixed = input.clone();
                mixed.flatten()[0] = 1.0;
                mixed.flatten()[mixed.numel() - 1] = -1.0;
                torch::Tensor mixed_output = leaky_relu_module->forward(mixed);
            }
        }
        
        // Test 4: Test with different shapes
        std::vector<std::vector<int64_t>> test_shapes = {
            {}, // scalar
            {0}, // empty 1D
            {1}, // single element
            {10}, // 1D
            {3, 4}, // 2D
            {2, 3, 4}, // 3D
            {2, 2, 2, 2}, // 4D
            {1, 1, 1, 1, 1} // 5D with all dims = 1
        };
        
        for (const auto& shape : test_shapes)
        {
            try
            {
                torch::Tensor test_tensor;
                if (shape.empty())
                {
                    test_tensor = torch::randn({}, input.options());
                }
                else if (shape[0] == 0)
                {
                    test_tensor = torch::empty(shape, input.options());
                }
                else
                {
                    test_tensor = torch::randn(shape, input.options());
                }
                
                torch::Tensor shape_output = leaky_relu_module->forward(test_tensor);
                
                // Verify output shape matches input shape
                if (shape_output.sizes() != test_tensor.sizes())
                {
                    std::cerr << "Shape mismatch for shape test" << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                // Some shapes might not be valid for certain dtypes, continue
                continue;
            }
        }
        
        // Test 5: Test gradient computation if tensor requires grad
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble)
        {
            torch::Tensor grad_input = input.clone().requires_grad_(true);
            torch::Tensor grad_output = torch::nn::functional::leaky_relu(
                grad_input,
                torch::nn::functional::LeakyReLUFuncOptions()
                    .negative_slope(negative_slope)
                    .inplace(false)
            );
            
            if (grad_output.numel() > 0)
            {
                // Compute gradient
                torch::Tensor grad_sum = grad_output.sum();
                grad_sum.backward();
                
                // Check gradient exists
                if (grad_input.grad().defined())
                {
                    // Gradient should be 1 for positive inputs and negative_slope for negative inputs
                    torch::Tensor expected_grad = torch::where(
                        grad_input > 0,
                        torch::ones_like(grad_input),
                        torch::full_like(grad_input, negative_slope)
                    );
                    
                    // Only compare if negative_slope is finite
                    if (std::isfinite(negative_slope))
                    {
                        bool grads_match = torch::allclose(grad_input.grad(), expected_grad, 1e-5, 1e-8);
                        if (!grads_match && grad_input.numel() < 10)
                        {
                            // For small tensors, log the mismatch
                            std::cerr << "Gradient mismatch detected" << std::endl;
                        }
                    }
                }
            }
        }
        
        // Test 6: Test with special float values if applicable
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble)
        {
            std::vector<float> special_values = {
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f,
                -0.0f,
                std::numeric_limits<float>::min(),
                std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max()
            };
            
            for (float val : special_values)
            {
                torch::Tensor special_tensor = torch::full({1}, val, input.options());
                torch::Tensor special_output = leaky_relu_module->forward(special_tensor);
                
                // Just ensure it doesn't crash
            }
        }
        
        // Test 7: Batch processing
        if (input.dim() >= 2 && input.size(0) > 1)
        {
            // Process each batch element separately and compare with batch processing
            std::vector<torch::Tensor> batch_outputs;
            for (int64_t i = 0; i < input.size(0); ++i)
            {
                torch::Tensor batch_elem = input[i];
                batch_outputs.push_back(leaky_relu_module->forward(batch_elem));
            }
            
            // Stack and compare
            torch::Tensor stacked = torch::stack(batch_outputs);
            torch::Tensor batch_output = leaky_relu_module->forward(input);
            
            if (!torch::allclose(stacked, batch_output, 1e-5, 1e-8))
            {
                std::cerr << "Batch processing inconsistency detected" << std::endl;
            }
        }
        
        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors
        std::cerr << "PyTorch error: " << e.what() << std::endl;
        return 0; // Continue fuzzing
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // Discard the input
    }
    catch (...)
    {
        std::cerr << "Unknown exception caught" << std::endl;
        return -1;
    }
}