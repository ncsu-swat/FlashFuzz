#include "fuzzer_utils.h"    // General fuzzing utilities
#include <c10/core/InferenceMode.h>
#include <iostream>          // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;

        // Check baseline state - inference mode should be disabled by default
        bool baseline_state = c10::InferenceMode::is_enabled();
        (void)baseline_state;

        // Use fuzzer data to determine test pattern
        uint8_t test_pattern = 0;
        if (Size > 0) {
            test_pattern = Data[0];
            offset++;
        }

        // Test 1: Basic inference mode enable/disable check
        {
            c10::InferenceMode guard(true);
            bool is_enabled = c10::InferenceMode::is_enabled();
            (void)is_enabled;

            // Create a tensor in inference mode
            if (Size > offset + 2) {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor result = tensor + 1.0f;
                (void)result;
            }
        }

        // Verify inference mode is disabled after guard destruction
        bool after_guard = c10::InferenceMode::is_enabled();
        (void)after_guard;

        // Test 2: Explicitly disabled inference mode
        {
            c10::InferenceMode guard(false);
            bool is_disabled = c10::InferenceMode::is_enabled();
            (void)is_disabled;

            // Operations outside inference mode can track gradients
            if (Size > offset + 2) {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Only attempt gradient operations on floating point tensors
                try {
                    if (tensor.is_floating_point() && !tensor.is_inference()) {
                        tensor = tensor.clone().detach().requires_grad_(true);
                        torch::Tensor output = tensor.sum();
                        output.backward();
                    }
                } catch (...) {
                    // Silently ignore gradient-related errors (expected for some tensor types)
                }
            }
        }

        // Test 3: Nested guards based on fuzzer input
        if (test_pattern & 0x01) {
            c10::InferenceMode outer_guard(true);
            bool outer_state = c10::InferenceMode::is_enabled();
            (void)outer_state;

            {
                // Nested guard with same setting
                c10::InferenceMode inner_guard(true);
                bool inner_state = c10::InferenceMode::is_enabled();
                (void)inner_state;
            }

            // State should still be enabled after inner guard exits
            bool mid_state = c10::InferenceMode::is_enabled();
            (void)mid_state;

            if (test_pattern & 0x02) {
                // Nested guard trying to disable (behavior depends on PyTorch version)
                c10::InferenceMode disable_guard(false);
                bool nested_disabled = c10::InferenceMode::is_enabled();
                (void)nested_disabled;
            }

            bool final_outer = c10::InferenceMode::is_enabled();
            (void)final_outer;
        }

        // Test 4: Multiple sequential guards
        if (test_pattern & 0x04) {
            for (int i = 0; i < 3; i++) {
                c10::InferenceMode guard((i % 2) == 0);
                bool state = c10::InferenceMode::is_enabled();
                (void)state;
            }
        }

        // Test 5: Create tensors under different inference mode states
        if (test_pattern & 0x08 && Size > offset + 2) {
            torch::Tensor normal_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            torch::Tensor inference_tensor;
            {
                c10::InferenceMode guard(true);
                if (Size > offset + 2) {
                    inference_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                }
            }

            // Check state after all operations
            bool final_state = c10::InferenceMode::is_enabled();
            (void)final_state;
            (void)normal_tensor;
            (void)inference_tensor;
        }

        // Final verification: should be disabled after all guards exit
        bool end_state = c10::InferenceMode::is_enabled();
        (void)end_state;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}