#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for mode flag
        if (Size < 1) {
            return 0;
        }
        
        // Extract mode flag from fuzzer input
        bool mode = (Data[offset] % 2) == 1;
        offset++;
        
        // Create a tensor with requires_grad=true for testing
        auto x = torch::ones({2, 3}, torch::TensorOptions().requires_grad(true));
        
        // Test 1: Basic inference_mode context manager usage
        {
            torch::InferenceMode guard(mode);
            
            // Perform operations inside inference mode
            auto y = x * x;
            auto z = y + 1;
            auto w = z.sum();
            
            // Test tensor properties under inference mode
            if (mode) {
                // In inference mode, tensors should not require grad
                if (y.requires_grad()) {
                    std::cout << "Unexpected: tensor requires grad in inference mode" << std::endl;
                }
            }
            
            // Test various operations that might behave differently in inference mode
            auto reshaped = z.reshape({-1});
            auto sliced = z.slice(0, 0, 1);
            auto transposed = z.transpose(0, 1);
        }
        
        // Test 2: Nested inference mode contexts
        {
            torch::InferenceMode outer_guard(mode);
            auto y1 = x * 2;
            
            {
                torch::InferenceMode inner_guard(!mode);
                auto y2 = x * 3;
                auto combined = y1 + y2;
            }
            
            auto y3 = x * 4;
        }
        
        // Test 3: Test with different tensor types and operations
        if (offset < Size) {
            // Create different tensor types based on remaining input
            uint8_t tensor_type = Data[offset % Size];
            
            torch::Tensor test_tensor;
            switch (tensor_type % 4) {
                case 0:
                    test_tensor = torch::randn({3, 3}, torch::TensorOptions().requires_grad(true));
                    break;
                case 1:
                    test_tensor = torch::zeros({2, 2, 2}, torch::TensorOptions().requires_grad(true));
                    break;
                case 2:
                    test_tensor = torch::eye(4, torch::TensorOptions().requires_grad(true));
                    break;
                case 3:
                    test_tensor = torch::arange(10, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat));
                    break;
            }
            
            torch::InferenceMode guard(mode);
            
            // Test various operations
            auto result1 = test_tensor.pow(2);
            auto result2 = torch::relu(test_tensor);
            auto result3 = torch::sigmoid(test_tensor);
            auto result4 = test_tensor.mean();
            
            // Test view operations
            if (test_tensor.numel() >= 4) {
                auto viewed = test_tensor.view({-1});
                auto selected = test_tensor.select(0, 0);
            }
        }
        
        // Test 4: Test inference mode with autograd operations
        {
            auto input = torch::randn({2, 2}, torch::TensorOptions().requires_grad(true));
            
            torch::InferenceMode guard(mode);
            
            // These operations should work but behave differently in inference mode
            auto output = input * input;
            auto loss = output.sum();
            
            // Test that we can still do forward operations
            auto grad_output = torch::ones_like(loss);
        }
        
        // Test 5: Test with different data types
        if (offset + 1 < Size) {
            uint8_t dtype_choice = Data[(offset + 1) % Size];
            
            torch::ScalarType dtype;
            switch (dtype_choice % 6) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kBool; break;
                case 5: dtype = torch::kFloat16; break;
            }
            
            try {
                auto typed_tensor = torch::ones({2, 2}, torch::TensorOptions().dtype(dtype).requires_grad(dtype.isFloatingType()));
                
                torch::InferenceMode guard(mode);
                
                if (dtype.isFloatingType()) {
                    auto result = typed_tensor * 2.0;
                    auto sum_result = result.sum();
                } else {
                    auto result = typed_tensor * 2;
                    auto sum_result = result.sum();
                }
            } catch (const std::exception& e) {
                // Some dtype combinations might not support requires_grad
                // This is expected behavior
            }
        }
        
        // Test 6: Test multiple sequential inference mode contexts
        for (int i = 0; i < 3; i++) {
            bool current_mode = ((Data[0] + i) % 2) == 1;
            torch::InferenceMode guard(current_mode);
            
            auto temp_tensor = torch::randn({2, 2}, torch::TensorOptions().requires_grad(true));
            auto temp_result = temp_tensor.pow(i + 1);
        }
        
        // Test 7: Test with empty tensors and edge cases
        {
            torch::InferenceMode guard(mode);
            
            // Empty tensor
            auto empty_tensor = torch::empty({0}, torch::TensorOptions().requires_grad(true));
            
            // Single element tensor
            auto single_tensor = torch::tensor(5.0, torch::TensorOptions().requires_grad(true));
            auto single_result = single_tensor * 2;
            
            // Large tensor (if we have enough input data)
            if (Size > 10) {
                size_t dim_size = (Data[Size - 1] % 10) + 1;
                auto large_tensor = torch::ones({dim_size, dim_size}, torch::TensorOptions().requires_grad(true));
                auto large_result = large_tensor.sum();
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}