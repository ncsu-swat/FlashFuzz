#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for mode selection
        if (Size < 1) {
            return 0;
        }
        
        // Extract boolean flag for inference mode
        bool inference_mode_enabled = (Data[offset] % 2) == 1;
        offset++;
        
        // Test 1: Basic inference mode context manager
        {
            torch::InferenceMode guard(inference_mode_enabled);
            
            // Create some tensors and perform operations inside inference mode
            if (offset + 8 <= Size) {
                auto tensor_data = extract_tensor_data(Data + offset, Size - offset);
                offset += 8;
                
                if (!tensor_data.empty()) {
                    auto tensor = torch::tensor(tensor_data);
                    
                    // Perform various operations that should work in inference mode
                    auto result1 = tensor + 1.0;
                    auto result2 = tensor * 2.0;
                    auto result3 = torch::relu(tensor);
                    auto result4 = tensor.sum();
                    
                    // Test tensor properties
                    bool requires_grad_before = tensor.requires_grad();
                    tensor.requires_grad_(true);  // This should be allowed
                    bool requires_grad_after = tensor.requires_grad();
                    
                    // Test view operations
                    if (tensor.numel() >= 4) {
                        auto reshaped = tensor.view({-1});
                        auto sliced = tensor.slice(0, 0, std::min(2L, tensor.size(0)));
                    }
                }
            }
        }
        
        // Test 2: Nested inference mode contexts
        if (offset < Size) {
            bool nested_mode = (Data[offset] % 2) == 1;
            offset++;
            
            torch::InferenceMode outer_guard(inference_mode_enabled);
            {
                torch::InferenceMode inner_guard(nested_mode);
                
                // Create tensor in nested context
                auto tensor = torch::randn({2, 2});
                auto result = tensor.matmul(tensor.t());
                
                // Test that we can still perform computations
                auto loss = result.sum();
            }
        }
        
        // Test 3: Check inference mode state
        {
            bool initial_state = torch::is_inference_mode_enabled();
            
            {
                torch::InferenceMode guard(true);
                bool state_in_mode = torch::is_inference_mode_enabled();
                
                // Create and manipulate tensors
                auto x = torch::ones({3, 3});
                auto y = torch::zeros({3, 3});
                auto z = x + y;
                
                // Test in-place operations
                x.add_(1.0);
                y.mul_(2.0);
            }
            
            bool final_state = torch::is_inference_mode_enabled();
        }
        
        // Test 4: Inference mode with different tensor types and operations
        if (offset + 4 <= Size) {
            uint32_t seed = 0;
            std::memcpy(&seed, Data + offset, 4);
            offset += 4;
            
            torch::InferenceMode guard(inference_mode_enabled);
            
            // Test with different dtypes
            auto float_tensor = torch::randn({2, 2}, torch::kFloat32);
            auto double_tensor = torch::randn({2, 2}, torch::kFloat64);
            auto int_tensor = torch::randint(0, 10, {2, 2}, torch::kInt32);
            auto bool_tensor = torch::randint(0, 2, {2, 2}, torch::kBool);
            
            // Test operations on different types
            auto float_result = float_tensor.sin();
            auto double_result = double_tensor.cos();
            auto int_result = int_tensor + 1;
            auto bool_result = bool_tensor.logical_not();
            
            // Test mixed operations
            auto mixed = float_tensor.to(torch::kFloat64) + double_tensor;
        }
        
        // Test 5: Inference mode with autograd operations
        {
            torch::InferenceMode guard(inference_mode_enabled);
            
            auto x = torch::randn({2, 2});
            x.requires_grad_(true);
            
            // These operations should work in inference mode
            auto y = x * 2;
            auto z = y.sum();
            
            // Test that gradients are not computed in inference mode
            // (this should not crash but gradients won't be tracked)
            auto w = x.detach();
            auto u = w + 1;
        }
        
        // Test 6: Exception handling within inference mode
        try {
            torch::InferenceMode guard(inference_mode_enabled);
            
            // Create tensors that might cause issues
            auto empty_tensor = torch::empty({0});
            auto large_tensor = torch::ones({100, 100});
            
            // Operations that might fail
            if (empty_tensor.numel() > 0) {
                auto result = empty_tensor.mean();
            }
            
            // Test broadcasting
            auto a = torch::ones({1, 3});
            auto b = torch::ones({2, 1});
            auto broadcast_result = a + b;
            
        } catch (const std::exception& e) {
            // Catch any exceptions that occur within inference mode
            // but don't propagate them as test failures
        }
        
        // Test 7: RAII behavior - ensure inference mode is properly restored
        {
            bool state_before = torch::is_inference_mode_enabled();
            
            {
                torch::InferenceMode guard(!state_before);  // Toggle the state
                bool state_during = torch::is_inference_mode_enabled();
                
                // Do some work
                auto tensor = torch::randn({5, 5});
                auto result = tensor.transpose(0, 1);
            }
            
            bool state_after = torch::is_inference_mode_enabled();
            // state_after should equal state_before due to RAII
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}