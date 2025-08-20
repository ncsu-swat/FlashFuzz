#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/monitor/events.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if vitals are enabled before any operations
        bool vitals_enabled_before = torch::monitor::vitals_enabled();
        
        // Create a tensor to use in operations
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations that might be monitored by vitals
            torch::Tensor result = tensor + 1;
            
            // Try to enable vitals if there's more data
            if (offset < Size) {
                bool enable_vitals = Data[offset++] % 2 == 0;
                torch::monitor::set_vitals_enabled(enable_vitals);
            }
            
            // Check if vitals are enabled after setting
            bool vitals_enabled_after = torch::monitor::vitals_enabled();
            
            // Perform more operations with vitals potentially enabled
            torch::Tensor another_result = result * 2;
            
            // Try to disable vitals if there's more data
            if (offset < Size) {
                bool disable_vitals = Data[offset++] % 2 == 0;
                torch::monitor::set_vitals_enabled(!disable_vitals);
            }
            
            // Final check of vitals state
            bool vitals_enabled_final = torch::monitor::vitals_enabled();
            
            // Use the results to prevent optimization
            if (another_result.numel() > 0) {
                auto sum = another_result.sum();
            }
        } else {
            // Even with no data, we can still test the vitals API
            bool initial_state = torch::monitor::vitals_enabled();
            torch::monitor::set_vitals_enabled(!initial_state);
            bool new_state = torch::monitor::vitals_enabled();
            torch::monitor::set_vitals_enabled(initial_state);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}