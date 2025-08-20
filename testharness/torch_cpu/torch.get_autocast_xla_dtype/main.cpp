#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/autocast_mode.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the enabled flag
        if (Size < 1) {
            return 0;
        }
        
        // Parse a boolean flag for whether autocast is enabled
        bool enabled = Data[offset++] & 0x1;
        
        // Try to get the autocast XLA dtype
        torch::ScalarType dtype = torch::autocast::get_autocast_xla_dtype(enabled);
        
        // Verify the result is a valid dtype
        if (dtype != torch::kFloat && dtype != torch::kBFloat16) {
            // This is not expected behavior, so we'll keep the input
            return 1;
        }
        
        // If we have more data, try with a tensor
        if (offset < Size) {
            // Create a tensor from the remaining data
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Get the autocast XLA dtype again
            dtype = torch::autocast::get_autocast_xla_dtype(enabled);
            
            // Try to cast the tensor to the autocast dtype
            torch::Tensor casted_tensor = tensor.to(dtype);
            
            // Verify the casted tensor has the expected dtype
            if (casted_tensor.scalar_type() != dtype) {
                return 1;
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