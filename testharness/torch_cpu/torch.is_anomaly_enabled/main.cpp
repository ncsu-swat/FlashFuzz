#include "fuzzer_utils.h"
#include <iostream>
#include <torch/csrc/autograd/anomaly_mode.h>

// target API keyword: torch.is_anomaly_enabled

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Save original anomaly detection state
        bool original_anomaly_enabled = torch::autograd::AnomalyMode::is_enabled();
        bool original_check_nan = torch::autograd::AnomalyMode::should_check_nan();
        
        // Toggle anomaly detection based on input data
        if (Size > offset)
        {
            bool enable_anomaly = Data[offset++] & 0x1;
            bool enable_check_nan = original_check_nan;
            if (Size > offset)
            {
                enable_check_nan = Data[offset++] & 0x1;
            }
            
            torch::autograd::AnomalyMode::set_enabled(enable_anomaly, enable_check_nan);
            
            // Verify that the setting was applied
            bool new_state = torch::autograd::AnomalyMode::is_enabled();
            bool new_check_nan = torch::autograd::AnomalyMode::should_check_nan();
            
            if (new_state != enable_anomaly)
            {
                throw std::runtime_error("Anomaly detection state did not change as expected");
            }
            if (new_check_nan != enable_check_nan)
            {
                throw std::runtime_error("Check NaN state did not change as expected");
            }
        }
        
        // Create a tensor that requires grad to test anomaly detection
        if (Size > offset)
        {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Skip if tensor is empty
            if (tensor.numel() == 0)
            {
                torch::autograd::AnomalyMode::set_enabled(original_anomaly_enabled, original_check_nan);
                return 0;
            }
            
            // Ensure tensor is floating point and requires gradients
            if (!tensor.is_floating_point())
            {
                tensor = tensor.to(torch::kFloat32);
            }
            tensor = tensor.detach().requires_grad_(true);
            
            // Perform some operations that might trigger anomaly detection
            auto result = tensor * 2.0;
            
            // Check if anomaly detection is still in the expected state
            bool current_state = torch::autograd::AnomalyMode::is_enabled();
            
            // Try to trigger backward with potential NaN/Inf
            if (Size > offset && (Data[offset++] & 0x1))
            {
                try
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
                catch (...)
                {
                    // Anomaly detection may throw - this is expected behavior
                }
            }
            else
            {
                try
                {
                    result.sum().backward();
                }
                catch (...)
                {
                    // Backward may fail for various reasons
                }
            }
        }
        
        // Test toggling states multiple times
        if (Size > offset)
        {
            int num_toggles = (Data[offset++] % 4) + 1;
            for (int i = 0; i < num_toggles && offset < Size; i++)
            {
                bool state = Data[offset++] & 0x1;
                bool check_nan = (offset < Size) ? (Data[offset++] & 0x1) : false;
                torch::autograd::AnomalyMode::set_enabled(state, check_nan);
                
                // Verify state after each toggle
                (void)torch::autograd::AnomalyMode::is_enabled();
                (void)torch::autograd::AnomalyMode::should_check_nan();
            }
        }
        
        // Reset anomaly detection to its original state
        torch::autograd::AnomalyMode::set_enabled(original_anomaly_enabled, original_check_nan);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}