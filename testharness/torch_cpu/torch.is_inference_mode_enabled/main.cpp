#include "fuzzer_utils.h"    // General fuzzing utilities
#include <c10/core/InferenceMode.h>
#include <iostream>          // For cerr
#include <tuple>             // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

        // Check if inference mode is enabled (torch.is_inference_mode_enabled)
        bool is_enabled = c10::InferenceMode::is_enabled();

        // Try with inference mode enabled
        {
            c10::InferenceMode guard;
            bool is_enabled_in_guard = c10::InferenceMode::is_enabled();

            // Create a tensor and perform some operations to ensure inference mode works
            if (Size > offset + 2) {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

                // Perform some operations on the tensor
                torch::Tensor result = tensor + 1;

                // Check if gradients are being tracked (should be false in inference mode)
                bool requires_grad = result.requires_grad();
                (void)requires_grad;
            }
        }

        // Check if inference mode is disabled after guard is destroyed
        bool is_enabled_after_guard = c10::InferenceMode::is_enabled();

        // Try with inference mode disabled explicitly
        {
            c10::InferenceMode guard(false);
            bool is_disabled_in_guard = c10::InferenceMode::is_enabled();

            // Create another tensor and perform operations
            if (Size > offset + 2) {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

                // Perform some operations on the tensor
                torch::Tensor result = tensor * 2;

                // Check if gradients can be tracked
                if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble) {
                    tensor.set_requires_grad(true);
                    torch::Tensor output = tensor.sum();
                    output.backward();
                }
                (void)is_disabled_in_guard;
            }
        }

        // Nested inference mode guards
        {
            c10::InferenceMode outer_guard;
            bool is_enabled_outer = c10::InferenceMode::is_enabled();

            {
                c10::InferenceMode inner_guard;
                bool is_enabled_inner = c10::InferenceMode::is_enabled();
                (void)is_enabled_inner;
            }

            bool is_still_enabled = c10::InferenceMode::is_enabled();

            {
                c10::InferenceMode inner_guard(false);
                bool is_disabled_inner = c10::InferenceMode::is_enabled();
                (void)is_disabled_inner;
            }

            bool is_enabled_after_inner = c10::InferenceMode::is_enabled();
            (void)is_enabled_outer;
            (void)is_still_enabled;
            (void)is_enabled_after_inner;
        }

        // Final check after all guards are destroyed
        bool final_state = c10::InferenceMode::is_enabled();
        (void)is_enabled;
        (void)is_enabled_after_guard;
        (void)final_state;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
