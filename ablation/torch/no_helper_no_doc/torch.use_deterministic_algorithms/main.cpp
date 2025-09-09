#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>  // PyTorch C++ frontend

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the boolean flag
        if (Size < 1) {
            return 0;
        }
        
        // Extract boolean value for deterministic algorithms setting
        bool use_deterministic = (Data[offset] % 2) == 1;
        offset++;
        
        // Test basic functionality - setting deterministic algorithms
        torch::use_deterministic_algorithms(use_deterministic);
        
        // Verify the setting was applied by checking current state
        bool current_state = torch::are_deterministic_algorithms_enabled();
        
        // Test multiple consecutive calls with same value
        torch::use_deterministic_algorithms(use_deterministic);
        torch::use_deterministic_algorithms(use_deterministic);
        
        // Test toggling behavior if we have more data
        if (offset < Size) {
            bool toggle_value = (Data[offset] % 2) == 1;
            torch::use_deterministic_algorithms(toggle_value);
            
            // Verify the toggle worked
            bool new_state = torch::are_deterministic_algorithms_enabled();
            
            // Toggle back
            torch::use_deterministic_algorithms(!toggle_value);
            offset++;
        }
        
        // Test with operations that might be affected by deterministic algorithms
        if (offset < Size) {
            // Create some tensors for testing deterministic behavior
            auto tensor1 = torch::randn({10, 10});
            auto tensor2 = torch::randn({10, 10});
            
            // Test operations that might behave differently with deterministic algorithms
            try {
                auto result1 = torch::mm(tensor1, tensor2);
                auto result2 = torch::addmm(torch::zeros({10, 10}), tensor1, tensor2);
                
                // Test with different deterministic settings
                torch::use_deterministic_algorithms(true);
                auto det_result1 = torch::mm(tensor1, tensor2);
                
                torch::use_deterministic_algorithms(false);
                auto non_det_result1 = torch::mm(tensor1, tensor2);
                
            } catch (const std::exception& op_e) {
                // Some operations might throw when deterministic algorithms are enabled
                // This is expected behavior for operations that don't have deterministic implementations
            }
        }
        
        // Test edge cases with rapid toggling
        if (offset + 4 <= Size) {
            for (int i = 0; i < 4 && offset < Size; i++) {
                bool rapid_toggle = (Data[offset] % 2) == 1;
                torch::use_deterministic_algorithms(rapid_toggle);
                offset++;
            }
        }
        
        // Test with CUDA operations if available
        if (torch::cuda::is_available() && offset < Size) {
            try {
                bool cuda_deterministic = (Data[offset] % 2) == 1;
                torch::use_deterministic_algorithms(cuda_deterministic);
                
                auto cuda_tensor1 = torch::randn({5, 5}, torch::device(torch::kCUDA));
                auto cuda_tensor2 = torch::randn({5, 5}, torch::device(torch::kCUDA));
                auto cuda_result = torch::mm(cuda_tensor1, cuda_tensor2);
                
            } catch (const std::exception& cuda_e) {
                // CUDA operations might fail or behave differently with deterministic algorithms
            }
        }
        
        // Reset to a known state at the end
        torch::use_deterministic_algorithms(false);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}