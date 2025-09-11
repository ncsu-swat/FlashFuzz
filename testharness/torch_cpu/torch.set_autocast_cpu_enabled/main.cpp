#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the boolean flag
        if (Size < 1) {
            return 0;
        }
        
        // Extract a boolean value from the first byte
        bool enabled = Data[0] & 0x1;
        offset++;
        
        // Set autocast CPU enabled state
        at::autocast::set_autocast_cpu_enabled(enabled);
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations with autocast enabled/disabled
            at::autocast::AutocastMode guard(at::kCPU, enabled);
            
            // Try different operations that might be affected by autocast
            auto result1 = tensor + tensor;
            auto result2 = tensor * tensor;
            auto result3 = torch::matmul(tensor, tensor);
            
            // Try to force computation to verify autocast behavior
            result1.sum().item<float>();
            result2.sum().item<float>();
            
            // Try with different dtypes that might be affected by autocast
            if (offset < Size) {
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                auto mixed_result = tensor + tensor2;
                mixed_result.sum().item<float>();
            }
        }
        
        // Toggle the autocast state and test again
        at::autocast::set_autocast_cpu_enabled(!enabled);
        
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            at::autocast::AutocastMode guard(at::kCPU, !enabled);
            
            auto result = tensor + tensor;
            result.sum().item<float>();
        }
        
        // Reset to original state
        at::autocast::set_autocast_cpu_enabled(enabled);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
