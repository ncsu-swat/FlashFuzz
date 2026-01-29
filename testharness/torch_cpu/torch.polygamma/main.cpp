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
        
        // Need at least 2 bytes for the n parameter and the input tensor
        if (Size < 2) {
            return 0;
        }
        
        // Extract n parameter for polygamma (n=0 is digamma, n=1 is trigamma, etc.)
        // Limit to reasonable values (0-9) to avoid numerical instability with large n
        int64_t n = static_cast<int64_t>(Data[offset++]) % 10;
        
        // Create input tensor for polygamma
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (...) {
            return 0; // If tensor creation fails, discard the input
        }
        
        // polygamma requires floating point input
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Apply polygamma operation
        torch::Tensor result;
        try {
            result = torch::polygamma(n, input_tensor);
        } catch (...) {
            // Expected exceptions for invalid inputs (e.g., negative integers) are fine
            return 0;
        }
        
        // Try to access result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            // Force evaluation of the tensor
            auto item = result.flatten()[0].item<double>();
            (void)item; // Prevent unused variable warning
        }
        
        // Also test the out variant if we have enough data
        if (Size > 10) {
            try {
                torch::Tensor out_tensor = torch::empty_like(input_tensor);
                torch::polygamma_out(out_tensor, n, input_tensor);
                
                if (out_tensor.defined() && out_tensor.numel() > 0) {
                    auto out_item = out_tensor.flatten()[0].item<double>();
                    (void)out_item;
                }
            } catch (...) {
                // Out variant may fail for various reasons, that's fine
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