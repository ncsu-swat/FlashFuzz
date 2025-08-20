#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if inference mode is enabled
        bool is_enabled = torch::InferenceMode::is_enabled();
        
        // Try with inference mode enabled
        {
            torch::InferenceMode guard;
            bool is_enabled_in_guard = torch::InferenceMode::is_enabled();
            
            // Create a tensor and perform some operations to ensure inference mode works
            if (Size > offset + 2) {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform some operations on the tensor
                torch::Tensor result = tensor + 1;
                
                // Check if gradients are being tracked (should be false in inference mode)
                bool requires_grad = result.requires_grad();
            }
        }
        
        // Check if inference mode is disabled after guard is destroyed
        bool is_enabled_after_guard = torch::InferenceMode::is_enabled();
        
        // Try with inference mode disabled explicitly
        {
            torch::InferenceMode guard(false);
            bool is_disabled_in_guard = torch::InferenceMode::is_enabled();
            
            // Create another tensor and perform operations
            if (Size > offset + 2) {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform some operations on the tensor
                torch::Tensor result = tensor * 2;
                
                // Check if gradients can be tracked
                if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble) {
                    tensor.set_requires_grad(true);
                    torch::Tensor output = tensor.sum();
                    output.backward();
                }
            }
        }
        
        // Nested inference mode guards
        {
            torch::InferenceMode outer_guard;
            bool is_enabled_outer = torch::InferenceMode::is_enabled();
            
            {
                torch::InferenceMode inner_guard;
                bool is_enabled_inner = torch::InferenceMode::is_enabled();
            }
            
            bool is_still_enabled = torch::InferenceMode::is_enabled();
            
            {
                torch::InferenceMode inner_guard(false);
                bool is_disabled_inner = torch::InferenceMode::is_enabled();
            }
            
            bool is_enabled_after_inner = torch::InferenceMode::is_enabled();
        }
        
        // Final check after all guards are destroyed
        bool final_state = torch::InferenceMode::is_enabled();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}