#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least 2 bytes for mode and warn_only flags
        if (Size < 2) {
            return 0;
        }

        // Extract mode flag (deterministic on/off)
        bool mode = (Data[offset] % 2) == 1;
        offset++;

        // Extract warn_only flag
        bool warn_only = (Data[offset] % 2) == 1;
        offset++;

        // Test basic functionality - set deterministic algorithms
        torch::use_deterministic_algorithms(mode, warn_only);

        // Verify the setting took effect by checking current state
        bool current_mode = torch::are_deterministic_algorithms_enabled();
        
        // Test toggling the mode multiple times
        if (Size > offset) {
            for (size_t i = offset; i < Size && i < offset + 10; i++) {
                bool toggle_mode = (Data[i] % 2) == 1;
                bool toggle_warn = ((Data[i] >> 1) % 2) == 1;
                torch::use_deterministic_algorithms(toggle_mode, toggle_warn);
            }
        }

        // Test edge cases with different combinations
        torch::use_deterministic_algorithms(true, false);   // strict deterministic
        torch::use_deterministic_algorithms(true, true);    // deterministic with warnings
        torch::use_deterministic_algorithms(false, false);  // non-deterministic
        torch::use_deterministic_algorithms(false, true);   // non-deterministic with warnings

        // Test operations that might be affected by deterministic mode
        if (Size > offset + 4) {
            // Create some test tensors to trigger deterministic behavior checks
            auto tensor1 = torch::randn({2, 3}, torch::kFloat32);
            auto tensor2 = torch::randn({3, 4}, torch::kFloat32);
            
            // Test matrix multiplication (can be nondeterministic on CUDA)
            try {
                auto result = torch::mm(tensor1, tensor2);
            } catch (const std::exception& e) {
                // Expected if deterministic mode is on and no deterministic implementation
            }

            // Test with CUDA tensors if available
            if (torch::cuda::is_available()) {
                try {
                    auto cuda_tensor1 = tensor1.cuda();
                    auto cuda_tensor2 = tensor2.cuda();
                    auto cuda_result = torch::mm(cuda_tensor1, cuda_tensor2);
                } catch (const std::exception& e) {
                    // Expected if deterministic mode is on and CUBLAS_WORKSPACE_CONFIG not set
                }
            }
        }

        // Test with different tensor operations that have deterministic variants
        if (Size > offset + 8) {
            auto test_tensor = torch::randn({10, 10}, torch::kFloat32);
            
            try {
                // Test operations that might throw in deterministic mode
                auto sorted = torch::sort(test_tensor);
                auto topk = torch::topk(test_tensor.flatten(), 5);
            } catch (const std::exception& e) {
                // Some operations may throw in deterministic mode
            }
        }

        // Reset to a known state
        torch::use_deterministic_algorithms(false, false);

        // Test rapid mode switching to check for any race conditions or state issues
        for (int i = 0; i < 5; i++) {
            torch::use_deterministic_algorithms(i % 2 == 0, i % 3 == 0);
        }

        // Final verification
        bool final_state = torch::are_deterministic_algorithms_enabled();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}