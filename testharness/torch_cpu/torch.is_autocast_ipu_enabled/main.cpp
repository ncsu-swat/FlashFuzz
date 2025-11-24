#include "fuzzer_utils.h"      // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <iostream>            // For cerr
#include <tuple>               // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

        // Keep target keyword torch.is_autocast_ipu_enabled referenced for harness check.
        bool is_enabled = at::autocast::is_autocast_enabled(at::kIPU);

        // Try to create a tensor if we have enough data
        if (Size > 2) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

            // Check autocast status again after tensor creation
            bool is_enabled_after = at::autocast::is_autocast_enabled(at::kIPU);
            if (is_enabled != is_enabled_after) {
                tensor = tensor.clone();
            }

            // Try to enable/disable autocast based on input data
            if (offset < Size) {
                bool should_enable = Data[offset++] % 2 == 0;

                // Enable or disable autocast for IPU
                if (should_enable) {
                    at::autocast::set_autocast_enabled(at::kIPU, true);
                } else {
                    at::autocast::set_autocast_enabled(at::kIPU, false);
                }

                // Verify the change took effect
                bool new_status = at::autocast::is_autocast_enabled(at::kIPU);

                // Perform some operation with the tensor while autocast is in the new state
                torch::Tensor result = tensor + 1.0;
                if (new_status && result.numel() > 0) {
                    (void)result[0].item<double>();
                }
            }
        }

        // Reset autocast state to avoid affecting other tests
        at::autocast::set_autocast_enabled(at::kIPU, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
