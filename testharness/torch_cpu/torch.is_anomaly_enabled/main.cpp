#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if anomaly detection is enabled
        bool anomaly_enabled = torch::autograd::is_anomaly_enabled();
        
        // Toggle anomaly detection based on input data
        if (Size > offset)
        {
            bool enable_anomaly = Data[offset++] & 0x1;
            torch::autograd::set_anomaly_enabled(enable_anomaly);
            
            // Verify that the setting was applied
            bool new_state = torch::autograd::is_anomaly_enabled();
            if (new_state != enable_anomaly)
            {
                throw std::runtime_error("Anomaly detection state did not change as expected");
            }
        }
        
        // Create a tensor that requires grad to test anomaly detection
        if (Size > offset)
        {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make tensor require gradients
            tensor = tensor.detach().requires_grad_(true);
            
            // Perform some operations that might trigger anomaly detection
            auto result = tensor * 2.0;
            
            // Check if anomaly detection is still in the expected state
            bool final_state = torch::autograd::is_anomaly_enabled();
            
            // Try to trigger backward with potential NaN/Inf
            if (Size > offset && (Data[offset++] & 0x1))
            {
                // Create a tensor with potential problematic values
                auto options = torch::TensorOptions().dtype(tensor.dtype());
                auto grad_tensor = torch::ones_like(result);
                
                // Introduce some potential NaN/Inf values
                if (Size > offset && (Data[offset++] & 0x1))
                {
                    grad_tensor = grad_tensor / 0.0; // Potential Inf/NaN
                }
                
                // Backward pass might trigger anomaly detection
                result.backward(grad_tensor);
            }
            else if (result.numel() > 0)
            {
                // Regular backward
                result.sum().backward();
            }
        }
        
        // Reset anomaly detection to its original state
        torch::autograd::set_anomaly_enabled(anomaly_enabled);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}