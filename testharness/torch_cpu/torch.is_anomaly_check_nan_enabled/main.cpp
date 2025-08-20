#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Test the is_anomaly_enabled function
        bool is_enabled = torch::is_anomaly_enabled();
        
        // Try to toggle the setting
        if (Size > offset)
        {
            bool new_setting = Data[offset++] % 2 == 0;
            torch::set_anomaly_enabled(new_setting);
            
            // Verify the setting was changed
            bool updated_setting = torch::is_anomaly_enabled();
            if (updated_setting != new_setting)
            {
                throw std::runtime_error("Failed to update anomaly check setting");
            }
        }
        
        // Create a tensor that might contain NaN values
        if (Size > offset)
        {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Enable anomaly detection
            torch::set_anomaly_enabled(true);
            
            // Perform some operations that might produce NaN
            torch::Tensor result;
            try {
                // Division that might produce NaN
                result = tensor / tensor;
                
                // Log operation
                result = torch::log(tensor);
                
                // Square root operation
                result = torch::sqrt(tensor);
            } catch (const c10::Error& e) {
                // Expected exception when anomaly detection is enabled and NaN is produced
            }
            
            // Disable anomaly detection
            torch::set_anomaly_enabled(false);
            
            // Try the same operations with detection disabled
            try {
                result = tensor / tensor;
                result = torch::log(tensor);
                result = torch::sqrt(tensor);
            } catch (const c10::Error& e) {
                // Exception might still occur for other reasons
            }
        }
        
        // Reset to original state
        torch::set_anomaly_enabled(is_enabled);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}