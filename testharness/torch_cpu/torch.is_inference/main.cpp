#include "fuzzer_utils.h" // General fuzzing utilities
#include <c10/core/InferenceMode.h>
#include <iostream> // For cerr
#include <tuple>    // For std::get with lu_unpack result

// Target API: torch.is_inference

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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to test with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Initial inference state checks
        bool initial_mode = c10::InferenceMode::is_enabled();
        bool tensor_inference = torch::is_inference(tensor);
        (void)initial_mode;
        (void)tensor_inference;

        // Test enabling inference mode
        {
            c10::InferenceMode guard(true);
            bool enabled_mode = c10::InferenceMode::is_enabled();
            bool enabled_tensor = torch::is_inference(tensor);
            torch::Tensor result1 = tensor + 1;
            bool result1_inference = torch::is_inference(result1);
            (void)enabled_mode;
            (void)enabled_tensor;
            (void)result1_inference;
            (void)result1.sum();
        }

        // Test disabling inference mode
        {
            c10::InferenceMode guard(false);
            bool disabled_mode = c10::InferenceMode::is_enabled();
            bool disabled_tensor = torch::is_inference(tensor);
            torch::Tensor result2 = tensor + 2;
            bool result2_inference = torch::is_inference(result2);
            (void)disabled_mode;
            (void)disabled_tensor;
            (void)result2_inference;
            (void)result2.sum();
        }

        // Test nested inference modes
        {
            c10::InferenceMode outer(true);
            bool outer_mode = c10::InferenceMode::is_enabled();
            bool outer_tensor = torch::is_inference(tensor);
            
            {
                c10::InferenceMode inner(false);
                bool inner_mode = c10::InferenceMode::is_enabled();
                bool inner_tensor = torch::is_inference(tensor);
                (void)inner_mode;
                (void)inner_tensor;
            }
            
            bool after_inner_mode = c10::InferenceMode::is_enabled();
            bool after_inner_tensor = torch::is_inference(tensor);
            (void)outer_mode;
            (void)outer_tensor;
            (void)after_inner_mode;
            (void)after_inner_tensor;
        }
        
        // Test with tensor creation in inference mode
        {
            c10::InferenceMode creation_guard(true);
            torch::Tensor new_tensor = torch::ones_like(tensor);
            bool new_tensor_inference = torch::is_inference(new_tensor);
            (void)new_tensor_inference;
            (void)new_tensor.sum();
        }
        
        // Test with different tensor types - may fail due to shape mismatch
        if (offset + 1 < Size) {
            try {
                torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                c10::InferenceMode type_guard(true);
                torch::Tensor combined = tensor + another_tensor;
                bool combined_inference = torch::is_inference(combined);
                (void)combined_inference;
                (void)combined.sum();
            } catch (...) {
                // Silently ignore shape mismatches and other expected errors
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