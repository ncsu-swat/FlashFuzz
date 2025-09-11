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
        
        // Skip if we don't have enough data
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract eps value from the remaining data if available
        double eps = 1e-6; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is in a reasonable range
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-6;
            if (std::isnan(eps) || std::isinf(eps)) eps = 1e-6;
        }
        
        // Make a copy of the input tensor to preserve original data
        torch::Tensor tensor_copy = input_tensor.clone();
        
        // Apply logit_ in-place operation
        tensor_copy.logit_(eps);
        
        // Alternative: test the functional version as well if there's enough data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor result = torch::logit(input_tensor, eps);
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
