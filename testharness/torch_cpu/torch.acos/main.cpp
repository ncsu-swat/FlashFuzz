#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for acos operation
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply acos operation
        torch::Tensor result = torch::acos(input_tensor);
        
        // Force computation by accessing the result safely
        if (result.defined() && result.numel() > 0) {
            // Use item() on a single element to force computation without assuming shape
            volatile float first_element = result.flatten()[0].item<float>();
            (void)first_element;
        }
        
        // Try some variants of the operation
        if (offset < Size) {
            // Create another tensor with remaining data if possible
            torch::Tensor input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test in-place version
            try {
                torch::Tensor inplace_result = input_tensor2.clone();
                inplace_result.acos_();
            } catch (...) {
                // In-place may fail for certain dtypes, ignore silently
            }
            
            // Test out variant
            try {
                torch::Tensor out_tensor = torch::empty_like(input_tensor2);
                torch::acos_out(out_tensor, input_tensor2);
            } catch (...) {
                // Out variant may fail for certain dtypes, ignore silently
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}