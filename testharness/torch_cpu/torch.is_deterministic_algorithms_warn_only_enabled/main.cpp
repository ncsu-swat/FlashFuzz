#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    const char *target_api = "torch.is_deterministic_algorithms_warn_only_enabled";
    (void)target_api;

    // Save initial state
    const bool initial_det = at::globalContext().deterministicAlgorithms();
    const bool initial_warn = at::globalContext().deterministicAlgorithmsWarnOnly();

    try
    {
        size_t offset = 0;
        
        // Get boolean values from input data to configure deterministic state
        bool enable_det = false;
        bool enable_warn_only = false;
        
        if (offset < Size) {
            enable_det = Data[offset++] & 0x1;
        }
        if (offset < Size) {
            enable_warn_only = Data[offset++] & 0x1;
        }
        
        // Test various combinations of deterministic algorithm settings
        // Note: warn_only can only be true if deterministic is also true
        if (enable_det) {
            at::globalContext().setDeterministicAlgorithms(true, enable_warn_only);
        } else {
            // When deterministic is false, warn_only must also be false
            at::globalContext().setDeterministicAlgorithms(false, false);
        }
        
        // This is the target API - check if warn-only mode is enabled
        bool is_warn_only_enabled = at::globalContext().deterministicAlgorithmsWarnOnly();
        
        // Verify the state is consistent
        bool is_det_enabled = at::globalContext().deterministicAlgorithms();
        
        // If deterministic is false, warn_only should also be false
        if (!is_det_enabled && is_warn_only_enabled) {
            // This would be a bug in PyTorch
            std::cerr << "Inconsistent state: warn_only is true but deterministic is false" << std::endl;
        }
        
        // Test toggling the state multiple times based on input
        for (size_t i = offset; i < Size && i < offset + 4; i++) {
            bool new_det = Data[i] & 0x1;
            bool new_warn = (Data[i] >> 1) & 0x1;
            
            if (new_det) {
                at::globalContext().setDeterministicAlgorithms(true, new_warn);
            } else {
                at::globalContext().setDeterministicAlgorithms(false, false);
            }
            
            // Query the warn-only state after each change
            bool current_warn_only = at::globalContext().deterministicAlgorithmsWarnOnly();
            (void)current_warn_only;
        }
        
        // Restore initial state
        at::globalContext().setDeterministicAlgorithms(initial_det, initial_warn);
    }
    catch (const std::exception &e)
    {
        // Restore state before returning
        try {
            at::globalContext().setDeterministicAlgorithms(initial_det, initial_warn);
        } catch (...) {
            // Ignore errors during cleanup
        }
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}