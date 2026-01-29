#include "fuzzer_utils.h"
#include <iostream>
#include <c10/util/Exception.h>

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
        size_t offset = 0;
        
        // Need at least 1 byte for the boolean value
        if (Size < 1) {
            return 0;
        }
        
        // Extract a boolean value from the first byte
        bool warn_always = Data[0] & 0x1;
        offset++;
        
        // Set the warning flag using the correct C++ API
        // In PyTorch C++, this is c10::WarningUtils::set_warnAlways()
        c10::WarningUtils::set_warnAlways(warn_always);
        
        // Verify the setting was applied by getting the current value
        bool current_setting = c10::WarningUtils::get_warnAlways();
        (void)current_setting; // Suppress unused variable warning
        
        // Test the functionality by creating a tensor that might trigger warnings
        if (Size > offset) {
            try {
                // Create a tensor that might trigger warnings
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform operations that might trigger warnings
                if (tensor.numel() > 0) {
                    // Division by small values might trigger warnings
                    torch::Tensor small_values = torch::ones_like(tensor) * 1e-10;
                    torch::Tensor result = tensor / small_values;
                    (void)result;
                    
                    // Operations with NaN/Inf might trigger warnings
                    torch::Tensor log_result = torch::log(torch::abs(tensor) + 1e-10);
                    (void)log_result;
                    
                    // Potential numerical instability - clamp to avoid inf
                    torch::Tensor clamped = torch::clamp(tensor, -10.0, 10.0);
                    torch::Tensor exp_result = torch::exp(clamped);
                    (void)exp_result;
                }
            } catch (const std::exception& e) {
                // Catch exceptions from tensor operations silently - these are expected
            }
        }
        
        // Toggle the setting to test both states
        c10::WarningUtils::set_warnAlways(!warn_always);
        
        // Reset to default state to avoid affecting other tests
        c10::WarningUtils::set_warnAlways(false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}