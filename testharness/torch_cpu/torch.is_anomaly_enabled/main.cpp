#include "fuzzer_utils.h"                         // General fuzzing utilities
#include <iostream>                               // For cerr
#include <tuple>                                  // For std::get with lu_unpack result
#include <torch/csrc/autograd/anomaly_mode.h>     // AnomalyMode controls anomaly detection

// target API keyword: torch.is_anomaly_enabled

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Check if anomaly detection is enabled
        bool anomaly_enabled = torch::autograd::AnomalyMode::is_enabled();
        bool anomaly_check_nan = torch::autograd::AnomalyMode::should_check_nan();
        
        // Toggle anomaly detection based on input data
        if (Size > offset)
        {
            bool enable_anomaly = Data[offset++] & 0x1;
            bool enable_check_nan = anomaly_check_nan;
            if (Size > offset)
            {
                enable_check_nan = Data[offset++] & 0x1;
            }
            torch::autograd::AnomalyMode::set_enabled(enable_anomaly, enable_check_nan);
            
            // Verify that the setting was applied
            bool new_state = torch::autograd::AnomalyMode::is_enabled();
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
            bool final_state = torch::autograd::AnomalyMode::is_enabled();
            
            // Try to trigger backward with potential NaN/Inf
            if (Size > offset && (Data[offset++] & 0x1))
            {
                // Create a tensor with potential problematic values
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
                // Regular backward with a small perturbation when anomaly mode is off
                if (!final_state)
                {
                    result = result + 1;
                }
                result.sum().backward();
            }
        }
        
        // Reset anomaly detection to its original state
        torch::autograd::AnomalyMode::set_enabled(anomaly_enabled, anomaly_check_nan);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
