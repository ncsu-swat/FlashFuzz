#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <c10/util/Exception.h> // For WarningUtils

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
        
        // Set the warning flag
        c10::WarningUtils::set_warnAlways(warn_always);
        
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
                    
                    // Operations with NaN/Inf might trigger warnings
                    torch::Tensor log_result = torch::log(tensor);
                    
                    // Potential numerical instability
                    torch::Tensor exp_result = torch::exp(tensor * 100);
                }
            } catch (const std::exception& e) {
                // Catch exceptions from tensor operations but continue testing
            }
        }
        
        // Reset to default state to avoid affecting other tests
        c10::WarningUtils::set_warnAlways(false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
