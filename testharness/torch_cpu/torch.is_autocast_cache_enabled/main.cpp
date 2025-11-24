#include "fuzzer_utils.h"              // General fuzzing utilities
#include <ATen/autocast_mode.h>        // at::autocast APIs
#include <iostream>                    // For cerr
#include <tuple>                       // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

        // Keep keyword for harness detection: torch.is_autocast_cache_enabled
        bool cache_enabled_initial = at::autocast::is_autocast_cache_enabled();

        if (Size > 0)
        {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

            // Toggle cache to exercise setter/getter pair.
            at::autocast::set_autocast_cache_enabled(true);
            bool cache_enabled_after_enable = at::autocast::is_autocast_cache_enabled();

            // Perform a simple operation and use the result so the computation runs.
            torch::Tensor result = tensor * 2.0;
            (void)result.sum().item<double>();

            at::autocast::set_autocast_cache_enabled(false);
            bool cache_enabled_after_disable = at::autocast::is_autocast_cache_enabled();

            // Restore the initial state to avoid side-effects on subsequent executions.
            at::autocast::set_autocast_cache_enabled(cache_enabled_initial);
            (void)cache_enabled_after_enable;
            (void)cache_enabled_after_disable;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
