#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for the n parameter and the input tensor
        if (Size < 2) {
            return 0;
        }
        
        // Extract n parameter for polygamma (first derivative is n=1, second is n=2, etc.)
        int64_t n = static_cast<int64_t>(Data[offset++]) % 10; // Limit to reasonable values
        
        // Create input tensor for polygamma
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            return 0; // If tensor creation fails, discard the input
        }
        
        // Apply polygamma operation
        torch::Tensor result;
        try {
            result = torch::polygamma(n, input_tensor);
        } catch (const std::exception& e) {
            // Expected exceptions for invalid inputs are fine
            return 0;
        }
        
        // Try to access result to ensure computation is performed
        if (result.defined()) {
            auto sizes = result.sizes();
            auto numel = result.numel();
            
            // Force evaluation of the tensor
            if (numel > 0) {
                auto item = result.flatten()[0].item<double>();
                (void)item; // Prevent unused variable warning
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