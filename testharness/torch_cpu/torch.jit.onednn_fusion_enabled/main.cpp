#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <torch/csrc/jit/passes/onednn_graph_fuser.h>

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
        // Test isEnabled() - this always works
        bool current_state = torch::jit::RegisterLlgaFuseGraph::isEnabled();
        
        // If we have data, try to toggle the fusion setting
        // Note: setEnabled() requires MKLDNN builds, so it will throw
        // an exception in non-MKLDNN builds - this is expected behavior
        if (Size >= 1) {
            bool enable_fusion = (Data[0] % 2 == 0);
            
            try {
                // This will throw if MKLDNN is not built - that's expected
                torch::jit::RegisterLlgaFuseGraph::setEnabled(enable_fusion);
                
                // If we get here, MKLDNN is available, verify the setting
                bool new_state = torch::jit::RegisterLlgaFuseGraph::isEnabled();
                
                // Toggle back and forth if we have more data
                for (size_t i = 1; i < Size && i < 10; i++) {
                    bool toggle = (Data[i] % 2 == 0);
                    torch::jit::RegisterLlgaFuseGraph::setEnabled(toggle);
                    torch::jit::RegisterLlgaFuseGraph::isEnabled();
                }
                
                // Restore original state
                torch::jit::RegisterLlgaFuseGraph::setEnabled(current_state);
            } catch (const c10::Error& e) {
                // Expected when MKLDNN is not available - not an error
                // Just continue testing isEnabled()
            }
        }
        
        // Call isEnabled() multiple times based on input data
        // to increase coverage of the query path
        for (size_t i = 0; i < Size && i < 20; i++) {
            volatile bool state = torch::jit::RegisterLlgaFuseGraph::isEnabled();
            (void)state;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}