#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least 1 byte to determine if anomaly detection should be enabled
        if (Size < 1) {
            return 0;
        }
        
        // Extract a byte to determine if anomaly detection should be enabled
        bool enable_anomaly = (Data[offset++] % 2 == 0);
        
        // Also extract whether to enable check_nan (if we have another byte)
        bool check_nan = false;
        if (offset < Size) {
            check_nan = (Data[offset++] % 2 == 0);
        }
        
        // Set anomaly detection mode
        torch::autograd::AnomalyMode::set_enabled(enable_anomaly);
        
        // Verify the setting was applied
        bool is_enabled = torch::autograd::AnomalyMode::is_enabled();
        (void)is_enabled; // Use the result to prevent optimization
        
        // Create a tensor that requires gradients to test anomaly detection
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make tensor require gradients (need to use float type for gradients)
            tensor = tensor.to(torch::kFloat32).detach().requires_grad_(true);
            
            // Perform some operations that might trigger anomaly detection
            torch::Tensor result = tensor * 2;
            
            // Try to perform a backward pass
            if (result.numel() > 0) {
                try {
                    result.sum().backward();
                } catch (const c10::Error& e) {
                    // Expected behavior when anomalies are detected or gradient issues
                }
            }
            
            // Toggle anomaly detection
            torch::autograd::AnomalyMode::set_enabled(!enable_anomaly);
            
            // Create a NEW tensor for the second backward pass (fresh gradient graph)
            if (offset < Size) {
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                tensor2 = tensor2.to(torch::kFloat32).detach().requires_grad_(true);
                
                torch::Tensor another_result = tensor2.pow(2);
                try {
                    if (another_result.numel() > 0) {
                        another_result.sum().backward();
                    }
                } catch (const c10::Error& e) {
                    // Expected behavior when anomalies are detected
                }
            }
        }
        
        // Test toggling multiple times with different settings
        torch::autograd::AnomalyMode::set_enabled(true);
        torch::autograd::AnomalyMode::set_enabled(false);
        torch::autograd::AnomalyMode::set_enabled(enable_anomaly);
        
        // Reset to disabled state at end to not affect other tests
        torch::autograd::AnomalyMode::set_enabled(false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}