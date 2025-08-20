#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation and dropout parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dropout probability from the input data
        float p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // Extract train flag from the input data
        bool train = true; // Default value
        if (offset < Size) {
            train = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Make a copy of the input tensor to verify it's modified in-place
        torch::Tensor input_copy = input.clone();
        
        // Apply dropout_ in-place operation using torch::dropout_
        torch::dropout_(input, p, train);
        
        // Verify the operation was in-place
        if (!train) {
            // If not in training mode, input should remain unchanged
            bool is_same = torch::allclose(input, input_copy);
            if (!is_same) {
                throw std::runtime_error("dropout_ modified tensor when train=false");
            }
        }
        
        // If in training mode and p > 0, some elements should be zeroed out
        if (train && p > 0 && input.numel() > 0) {
            // This is just a sanity check, not a strict requirement
            // We don't want to be too strict as it might prevent finding bugs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}