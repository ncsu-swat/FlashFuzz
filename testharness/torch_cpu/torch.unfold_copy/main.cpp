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
        
        // Need at least a few bytes for tensor creation and unfold parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for unfold operation
        // We need at least 3 bytes for dimension, size, and step
        if (offset + 3 > Size) {
            return 0;
        }
        
        // Get dimension parameter (can be negative for indexing from the end)
        int64_t dimension = static_cast<int8_t>(Data[offset++]);
        
        // Get size parameter (should be positive)
        uint8_t size_param = Data[offset++];
        int64_t size = static_cast<int64_t>(size_param);
        
        // Get step parameter (can be any integer)
        int64_t step = static_cast<int8_t>(Data[offset++]);
        
        // Apply unfold_copy operation
        torch::Tensor result = torch::unfold_copy(input, dimension, size, step);
        
        // Optional: perform some basic validation on the result
        if (result.numel() > 0) {
            // Access some elements to ensure the tensor is valid
            auto flat = result.flatten();
            if (flat.numel() > 0) {
                flat[0].item<float>();
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
