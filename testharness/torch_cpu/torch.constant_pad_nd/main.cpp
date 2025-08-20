#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse pad values and dimensions
        // Need at least 1 byte for number of dimensions to pad
        if (offset + 1 >= Size) {
            return 0;
        }
        
        // Get number of dimensions to pad (between 1 and input.dim())
        uint8_t num_dims_to_pad = Data[offset++] % (std::max(static_cast<int64_t>(1), input.dim()) + 1);
        if (num_dims_to_pad == 0) {
            num_dims_to_pad = 1;
        }
        
        // Need 2 bytes per dimension (for padding before and after)
        if (offset + 2 * num_dims_to_pad + 1 >= Size) {
            return 0;
        }
        
        // Create padding vector
        std::vector<int64_t> pad;
        pad.reserve(2 * num_dims_to_pad);
        
        // Fill padding values
        for (int i = 0; i < num_dims_to_pad; i++) {
            // Get padding before and after for this dimension
            int8_t pad_before = static_cast<int8_t>(Data[offset++]);
            int8_t pad_after = static_cast<int8_t>(Data[offset++]);
            
            // Add padding values in reverse order (last dimension first)
            pad.insert(pad.begin(), pad_after);
            pad.insert(pad.begin(), pad_before);
        }
        
        // Get value to pad with
        float pad_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Apply constant_pad_nd operation
        torch::Tensor output = torch::constant_pad_nd(input, pad, pad_value);
        
        // Ensure the output is used to prevent optimization
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}