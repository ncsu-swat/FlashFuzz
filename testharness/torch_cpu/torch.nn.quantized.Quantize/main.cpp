#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse scale and zero_point from the remaining data
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive and not too extreme
            scale = std::abs(scale);
            if (scale < 1e-10) scale = 1e-10;
            if (scale > 1e10) scale = 1e10;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within valid range for quantized types
            zero_point = zero_point % 256;
        }
        
        // Parse dtype for quantization
        torch::ScalarType dtype = torch::kQInt8;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            if (dtype_selector % 2 == 0) {
                dtype = torch::kQInt8;
            } else {
                dtype = torch::kQUInt8;
            }
        }
        
        // Apply quantization using torch::quantize_per_tensor
        torch::Tensor output = torch::quantize_per_tensor(input_tensor, scale, zero_point, dtype);
        
        // Ensure the output is not optimized away
        volatile bool output_exists = output.defined();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}