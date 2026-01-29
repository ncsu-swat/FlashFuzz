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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply Hardswish using functional API
        // Hardswish(x) = x * ReLU6(x + 3) / 6
        torch::Tensor output = torch::hardswish(input);
        
        // Try hardswish_out version (writes result to pre-allocated tensor)
        if (input.is_floating_point()) {
            try {
                torch::Tensor out_tensor = torch::empty_like(input);
                torch::hardswish_out(out_tensor, input);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Try inplace version using hardswish_ function
        if (input.is_floating_point() && !input.requires_grad()) {
            try {
                torch::Tensor input_copy = input.clone();
                // Use the inplace variant through the at:: namespace
                at::hardswish_(input_copy);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test with different tensor configurations
        if (offset + 4 < Size) {
            // Create tensors with different properties based on fuzzer data
            bool requires_grad = Data[offset++] % 2 == 0;
            uint8_t dtype_selector = Data[offset++] % 3;
            
            try {
                torch::Tensor test_input;
                
                // Test with different dtypes
                switch (dtype_selector) {
                    case 0:
                        test_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        test_input = input.to(torch::kFloat64);
                        break;
                    default:
                        test_input = input.to(torch::kFloat16);
                        break;
                }
                
                if (requires_grad && test_input.is_floating_point()) {
                    test_input = test_input.clone().set_requires_grad(true);
                }
                
                torch::Tensor result = torch::hardswish(test_input);
                
                // If requires_grad, test backward pass
                if (test_input.requires_grad()) {
                    try {
                        result.sum().backward();
                    } catch (...) {
                        // Silently ignore gradient computation failures
                    }
                }
            } catch (...) {
                // Silently ignore expected failures (e.g., dtype conversion issues)
            }
        }
        
        // Test with various tensor shapes
        if (offset + 8 < Size) {
            try {
                // Create tensors with specific shapes derived from fuzzer data
                int64_t dim0 = (Data[offset++] % 8) + 1;  // 1-8
                int64_t dim1 = (Data[offset++] % 8) + 1;  // 1-8
                int64_t dim2 = (Data[offset++] % 8) + 1;  // 1-8
                
                // Test 1D tensor
                torch::Tensor tensor_1d = torch::randn({dim0});
                torch::Tensor out_1d = torch::hardswish(tensor_1d);
                
                // Test 2D tensor
                torch::Tensor tensor_2d = torch::randn({dim0, dim1});
                torch::Tensor out_2d = torch::hardswish(tensor_2d);
                
                // Test 3D tensor (typical for CNN activations)
                torch::Tensor tensor_3d = torch::randn({dim0, dim1, dim2});
                torch::Tensor out_3d = torch::hardswish(tensor_3d);
                
                // Test 4D tensor (batch of images)
                int64_t batch = (Data[offset++] % 4) + 1;
                torch::Tensor tensor_4d = torch::randn({batch, dim0, dim1, dim2});
                torch::Tensor out_4d = torch::hardswish(tensor_4d);
                
            } catch (...) {
                // Silently ignore shape-related failures
            }
        }
        
        // Test edge cases
        try {
            // Empty tensor
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_result = torch::hardswish(empty_tensor);
            
            // Single element tensor
            torch::Tensor single = torch::tensor({1.5f});
            torch::Tensor single_result = torch::hardswish(single);
            
            // Tensor with special values
            torch::Tensor special = torch::tensor({-4.0f, -3.0f, 0.0f, 3.0f, 4.0f});
            torch::Tensor special_result = torch::hardswish(special);
            
        } catch (...) {
            // Silently ignore edge case failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}