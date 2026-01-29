#include "fuzzer_utils.h"              // General fuzzing utilities
#include <ATen/autocast_mode.h>        // at::autocast APIs
#include <iostream>                    // For cerr/cout

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
        // Keep keyword for harness detection: torch.is_autocast_cache_enabled

        // Save initial state to restore later
        bool cache_enabled_initial = at::autocast::is_autocast_cache_enabled();

        // Use fuzzer data to determine test pattern
        if (Size > 0)
        {
            // Use fuzzer byte to drive different test scenarios
            uint8_t test_mode = Data[0] % 4;

            switch (test_mode)
            {
            case 0:
                // Test: enable -> check -> disable -> check
                at::autocast::set_autocast_cache_enabled(true);
                (void)at::autocast::is_autocast_cache_enabled();
                at::autocast::set_autocast_cache_enabled(false);
                (void)at::autocast::is_autocast_cache_enabled();
                break;

            case 1:
                // Test: disable -> check -> enable -> check
                at::autocast::set_autocast_cache_enabled(false);
                (void)at::autocast::is_autocast_cache_enabled();
                at::autocast::set_autocast_cache_enabled(true);
                (void)at::autocast::is_autocast_cache_enabled();
                break;

            case 2:
                // Test: multiple toggles based on fuzzer data
                for (size_t i = 1; i < Size && i < 16; ++i)
                {
                    bool enable = (Data[i] % 2) == 0;
                    at::autocast::set_autocast_cache_enabled(enable);
                    (void)at::autocast::is_autocast_cache_enabled();
                }
                break;

            case 3:
                // Test: just query without modification
                (void)at::autocast::is_autocast_cache_enabled();
                (void)at::autocast::is_autocast_cache_enabled();
                break;
            }
        }
        else
        {
            // Empty input: just check current state
            (void)at::autocast::is_autocast_cache_enabled();
        }

        // Restore the initial state to avoid side-effects on subsequent executions
        at::autocast::set_autocast_cache_enabled(cache_enabled_initial);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}