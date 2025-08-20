#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create a quantized tensor
        torch::Tensor scales;
        torch::Tensor zero_points;
        int64_t axis = 0;
        
        // Create scales tensor
        if (offset < Size) {
            scales = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create zero_points tensor with same shape as scales
        if (offset < Size) {
            zero_points = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to make zero_points match scales shape
            if (scales.defined() && zero_points.defined() && scales.sizes() != zero_points.sizes()) {
                if (scales.dim() > 0) {
                    zero_points = zero_points.reshape_as(scales);
                }
            }
        }
        
        // Get axis parameter
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&axis, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create a quantized tensor
        torch::Tensor quantized_tensor;
        
        if (scales.defined() && zero_points.defined()) {
            // Create a tensor to be quantized
            torch::Tensor input_tensor;
            if (offset < Size) {
                input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Create a default tensor if we've run out of data
                input_tensor = torch::ones({2, 3, 4});
            }
            
            // Try to create a quantized tensor
            if (input_tensor.defined()) {
                try {
                    // Get per-channel scales - only takes one argument
                    auto result = torch::q_per_channel_scales(scales);
                } catch (const c10::Error& e) {
                    // PyTorch specific exceptions are expected and part of testing
                }
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