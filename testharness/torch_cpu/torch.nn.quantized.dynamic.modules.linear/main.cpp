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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for the linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in_features from the input tensor if possible
        if (input.dim() >= 1) {
            in_features = input.size(-1);
        } else {
            // For scalar or empty tensor, use a small value
            in_features = 1;
        }
        
        // Get out_features from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            out_features = 1;
        }
        
        // Get bias parameter if there's data left
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create a regular linear module and then apply dynamic quantization
        torch::nn::Linear linear_module(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Apply the module to the input tensor
        torch::Tensor output;
        
        // Handle different input tensor dimensions
        if (input.dim() == 0) {
            // Scalar tensor - reshape to 1D
            output = linear_module(input.reshape({1, 1}));
        } else if (input.dim() == 1) {
            // 1D tensor - add batch dimension
            output = linear_module(input.unsqueeze(0));
        } else {
            // Multi-dimensional tensor
            output = linear_module(input);
        }
        
        // Apply dynamic quantization to the output
        torch::Tensor quantized_output = torch::quantize_per_tensor(
            output, 
            0.1, // scale
            0,   // zero_point
            torch::kQInt8
        );
        
        // Test some quantized tensor properties
        auto scale = quantized_output.q_scale();
        auto zero_point = quantized_output.q_zero_point();
        
        // Test dequantization
        torch::Tensor dequantized = quantized_output.dequantize();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}