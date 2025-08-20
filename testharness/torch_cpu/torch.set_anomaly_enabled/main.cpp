#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to determine if anomaly detection should be enabled
        if (Size < 1) {
            return 0;
        }
        
        // Extract a byte to determine if anomaly detection should be enabled
        bool enable_anomaly = (Data[offset++] % 2 == 0);
        
        // Set anomaly detection mode
        torch::autograd::AnomalyMode::set_enabled(enable_anomaly);
        
        // Create a tensor that requires gradients to test anomaly detection
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make tensor require gradients
            tensor = tensor.detach().requires_grad_(true);
            
            // Perform some operations that might trigger anomaly detection
            torch::Tensor result = tensor * 2;
            
            // Try to perform a backward pass
            if (result.numel() > 0 && result.dim() > 0) {
                try {
                    result.sum().backward();
                } catch (const c10::Error& e) {
                    // Expected behavior when anomalies are detected
                }
            }
            
            // Toggle anomaly detection again
            torch::autograd::AnomalyMode::set_enabled(!enable_anomaly);
            
            // Try another operation with the new setting
            if (offset < Size && tensor.numel() > 0) {
                torch::Tensor another_result = tensor.pow(2);
                try {
                    if (another_result.numel() > 0) {
                        another_result.sum().backward();
                    }
                } catch (const c10::Error& e) {
                    // Expected behavior when anomalies are detected
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}