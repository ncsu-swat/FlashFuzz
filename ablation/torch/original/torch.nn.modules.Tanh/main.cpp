#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <memory>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check - we need at least some bytes for tensor creation
        if (Size < 2) {
            // Not enough data to create even a minimal tensor
            return 0;
        }

        // Create Tanh module
        torch::nn::Tanh tanh_module;
        
        // Test 1: Basic forward pass with a single tensor
        try {
            auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply Tanh
            auto output = tanh_module->forward(input_tensor);
            
            // Verify output shape matches input shape
            if (output.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch after Tanh: input=" << input_tensor.sizes() 
                         << " output=" << output.sizes() << std::endl;
            }
            
            // Verify output is in range [-1, 1] for real-valued tensors
            if (!input_tensor.is_complex()) {
                auto min_val = output.min();
                auto max_val = output.max();
                if (min_val.item<double>() < -1.0 - 1e-6 || max_val.item<double>() > 1.0 + 1e-6) {
                    std::cerr << "Tanh output out of expected range [-1, 1]: min=" 
                             << min_val.item<double>() << " max=" << max_val.item<double>() << std::endl;
                }
            }
            
            // Test gradient computation if there's enough data
            if (offset < Size && input_tensor.dtype() == torch::kFloat || 
                input_tensor.dtype() == torch::kDouble || 
                input_tensor.dtype() == torch::kHalf || 
                input_tensor.dtype() == torch::kBFloat16) {
                
                input_tensor.requires_grad_(true);
                auto output_grad = tanh_module->forward(input_tensor);
                
                // Create a gradient tensor of same shape as output
                auto grad_output = torch::ones_like(output_grad);
                
                // Backward pass
                output_grad.backward(grad_output);
                
                // Check if gradient was computed
                if (input_tensor.grad().defined()) {
                    // Gradient of tanh(x) is 1 - tanh^2(x)
                    auto expected_grad = 1 - output_grad.detach() * output_grad.detach();
                    if (!torch::allclose(input_tensor.grad(), expected_grad, 1e-4, 1e-6)) {
                        std::cerr << "Gradient mismatch in Tanh backward pass" << std::endl;
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors - these are expected for some inputs
            return 0;
        } catch (const std::exception& e) {
            // Other errors during first tensor processing
            return 0;
        }
        
        // Test 2: Multiple tensors with different properties if we have more data
        while (offset + 2 < Size) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test with different tensor states
                if (Data[offset % Size] & 0x01) {
                    // Test with contiguous tensor
                    tensor = tensor.contiguous();
                }
                
                if (Data[offset % Size] & 0x02) {
                    // Test with non-contiguous tensor (transpose if 2D or higher)
                    if (tensor.dim() >= 2) {
                        tensor = tensor.transpose(0, 1);
                    }
                }
                
                if (Data[offset % Size] & 0x04) {
                    // Test with view if possible
                    if (tensor.numel() > 0) {
                        auto new_shape = std::vector<int64_t>{tensor.numel()};
                        tensor = tensor.view(new_shape);
                    }
                }
                
                // Apply Tanh
                auto result = tanh_module->forward(tensor);
                
                // Additional edge case testing
                if (Data[offset % Size] & 0x08) {
                    // Test with special values if float type
                    if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble) {
                        // Create tensor with special values
                        auto special_tensor = torch::tensor({
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN(),
                            0.0f, -0.0f, 1.0f, -1.0f
                        }, torch::dtype(tensor.dtype()));
                        
                        if (special_tensor.numel() <= tensor.numel() && tensor.numel() > 0) {
                            // Flatten tensor and set first few values to special values
                            auto flat_tensor = tensor.flatten();
                            flat_tensor.narrow(0, 0, special_tensor.numel()).copy_(special_tensor);
                            tensor = flat_tensor.view(tensor.sizes());
                            
                            // Apply Tanh to tensor with special values
                            auto special_result = tanh_module->forward(tensor);
                            
                            // tanh(inf) = 1, tanh(-inf) = -1, tanh(0) = 0
                            // We don't strictly verify NaN handling as it's implementation-defined
                        }
                    }
                }
                
                // Test in-place operation if supported
                if (Data[offset % Size] & 0x10) {
                    if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble) {
                        auto tensor_copy = tensor.clone();
                        tensor_copy.tanh_();  // In-place tanh
                        
                        // Compare with module output
                        if (!torch::allclose(tensor_copy, result, 1e-5, 1e-7)) {
                            std::cerr << "In-place tanh differs from module output" << std::endl;
                        }
                    }
                }
                
                // Test with zero-element tensors
                if (tensor.numel() == 0) {
                    // Should handle gracefully
                    auto empty_result = tanh_module->forward(tensor);
                    if (empty_result.numel() != 0) {
                        std::cerr << "Zero-element tensor produced non-zero output" << std::endl;
                    }
                }
                
            } catch (const c10::Error& e) {
                // Expected for some edge cases
                continue;
            } catch (const std::exception& e) {
                // Expected for some edge cases
                continue;
            }
        }
        
        // Test 3: Module state and parameters
        {
            // Tanh has no learnable parameters
            auto params = tanh_module->parameters();
            if (!params.empty()) {
                std::cerr << "Tanh module unexpectedly has parameters" << std::endl;
            }
            
            // Test module training/eval mode (shouldn't affect Tanh but test anyway)
            tanh_module->train();
            tanh_module->eval();
            
            // Test serialization if we have data left
            if (offset < Size) {
                try {
                    // Save and load the module
                    std::stringstream stream;
                    torch::save(tanh_module, stream);
                    
                    torch::nn::Tanh loaded_module;
                    torch::load(loaded_module, stream);
                    
                    // Test that loaded module works the same
                    if (offset + 2 < Size) {
                        auto test_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        auto orig_output = tanh_module->forward(test_tensor);
                        auto loaded_output = loaded_module->forward(test_tensor);
                        
                        if (!torch::allclose(orig_output, loaded_output, 1e-6, 1e-8)) {
                            std::cerr << "Loaded module produces different output" << std::endl;
                        }
                    }
                } catch (...) {
                    // Serialization might fail for some edge cases
                }
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}