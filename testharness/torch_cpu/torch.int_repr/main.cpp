#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // int_repr only works on quantized tensors, so we need to quantize the tensor first
        // We'll try different quantization schemes based on the input data
        
        // Get a byte from the input data to determine quantization parameters
        uint8_t quant_param = (offset < Size) ? Data[offset++] : 0;
        
        // Create a quantized tensor
        torch::Tensor quantized_tensor;
        
        try {
            // Choose quantization type based on input data
            if (quant_param % 3 == 0) {
                // Per tensor quantization
                double scale = 0.01 + (quant_param % 100) * 0.001;
                int64_t zero_point = quant_param % 256;
                
                // Try different quantized dtypes
                if (quant_param % 2 == 0) {
                    quantized_tensor = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
                } else {
                    quantized_tensor = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
                }
            } else if (quant_param % 3 == 1) {
                // Per channel quantization (only works on tensors with at least 1 dimension)
                if (input_tensor.dim() > 0) {
                    int64_t channel_dim = quant_param % std::max(static_cast<int64_t>(1), input_tensor.dim());
                    int64_t num_channels = input_tensor.size(channel_dim);
                    
                    // Create scales and zero_points
                    std::vector<double> scales(num_channels, 0.01);
                    std::vector<int64_t> zero_points(num_channels, 0);
                    
                    // Vary scales and zero_points based on input data
                    for (int64_t i = 0; i < num_channels && (offset + i < Size); i++) {
                        scales[i] = 0.01 + (Data[offset + i] % 100) * 0.001;
                        if (offset + num_channels + i < Size) {
                            zero_points[i] = Data[offset + num_channels + i] % 256;
                        }
                    }
                    
                    offset += 2 * num_channels;
                    
                    // Create per-channel quantized tensor
                    quantized_tensor = torch::quantize_per_channel(
                        input_tensor, 
                        torch::from_blob(scales.data(), {num_channels}, torch::kDouble).clone(),
                        torch::from_blob(zero_points.data(), {num_channels}, torch::kLong).clone(),
                        channel_dim,
                        torch::kQInt8
                    );
                } else {
                    // Fallback for scalar tensors
                    double scale = 0.01 + (quant_param % 100) * 0.001;
                    int64_t zero_point = quant_param % 256;
                    quantized_tensor = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
                }
            } else {
                // Another per tensor quantization with different parameters
                double scale = 0.1 + (quant_param % 10) * 0.01;
                int64_t zero_point = quant_param % 128;
                quantized_tensor = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
            }
            
            // Apply int_repr to the quantized tensor
            torch::Tensor int_repr_result = torch::int_repr(quantized_tensor);
            
            // Perform some operations on the result to ensure it's used
            auto sum = int_repr_result.sum();
            auto mean = int_repr_result.mean();
            
            // Check that int_repr returns a tensor with the same shape as the input
            if (int_repr_result.sizes() != quantized_tensor.sizes()) {
                throw std::runtime_error("int_repr result has different shape than input");
            }
            
            // Check that int_repr returns a tensor with integer dtype
            if (int_repr_result.dtype() != torch::kInt8 && int_repr_result.dtype() != torch::kUInt8 && 
                int_repr_result.dtype() != torch::kInt16 && int_repr_result.dtype() != torch::kInt32 && 
                int_repr_result.dtype() != torch::kInt64) {
                throw std::runtime_error("int_repr result is not an integer tensor");
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and should be caught
            // We don't need to do anything special here
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}