#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for quantization
        // We need scale and zero_point for quantization
        float scale = 0.1f;
        int64_t zero_point = 0;
        
        // If we have more data, use it to set scale and zero_point
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive and not too small or large
            scale = std::abs(scale);
            if (scale < 1e-10f) scale = 1e-10f;
            if (scale > 1e10f) scale = 1e10f;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within valid range for int8
            zero_point = std::max<int64_t>(std::min<int64_t>(zero_point, 127), -128);
        }
        
        // Create quantization parameters
        torch::ScalarType dtype = torch::kQInt8;
        if (offset < Size) {
            // Use one more byte to select between qint8 and quint8
            uint8_t dtype_selector = Data[offset++];
            dtype = (dtype_selector % 2 == 0) ? torch::kQInt8 : torch::kQUInt8;
            
            // Adjust zero_point range for quint8 if needed
            if (dtype == torch::kQUInt8) {
                zero_point = std::max<int64_t>(std::min<int64_t>(zero_point, 255), 0);
            }
        }
        
        // Apply quantization using torch::quantize_per_tensor
        torch::Tensor output = torch::quantize_per_tensor(input_tensor, scale, zero_point, dtype);
        
        // Try to access tensor properties to ensure computation happened
        auto sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try to get the quantization parameters from the output tensor
        auto qparams = output.q_scale();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
