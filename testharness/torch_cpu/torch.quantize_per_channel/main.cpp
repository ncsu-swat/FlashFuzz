#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty or scalar tensors - quantize_per_channel needs at least 1D
        if (input_tensor.dim() == 0 || input_tensor.numel() == 0) {
            return 0;
        }
        
        // Convert input to float (required for quantization)
        input_tensor = input_tensor.to(torch::kFloat);
        
        // Get axis parameter (must be valid for the tensor dimensions)
        int64_t axis = 0;
        if (offset < Size) {
            uint8_t axis_byte = Data[offset++];
            axis = static_cast<int64_t>(axis_byte % input_tensor.dim());
        }
        
        // Get the size along the quantization axis
        int64_t axis_size = input_tensor.size(axis);
        if (axis_size == 0) {
            return 0;
        }
        
        // Get dtype for quantization
        torch::ScalarType dtype = torch::kQUInt8;
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++];
            if (dtype_byte % 2 == 0) {
                dtype = torch::kQUInt8;
            } else {
                dtype = torch::kQInt8;
            }
        }
        
        // Create scales tensor - must be 1D float with size matching axis_size
        // Scales must be positive
        torch::Tensor scales = torch::empty({axis_size}, torch::kFloat);
        for (int64_t i = 0; i < axis_size; i++) {
            float scale_val = 0.01f; // default
            if (offset < Size) {
                // Generate scale in range (0, 10]
                scale_val = (static_cast<float>(Data[offset++]) / 255.0f) * 9.99f + 0.01f;
            }
            scales[i] = scale_val;
        }
        
        // Create zero_points tensor - must be 1D int64 with size matching axis_size
        // For kQUInt8: range [0, 255], for kQInt8: range [-128, 127]
        torch::Tensor zero_points = torch::empty({axis_size}, torch::kLong);
        for (int64_t i = 0; i < axis_size; i++) {
            int64_t zp_val = 0;
            if (offset < Size) {
                if (dtype == torch::kQUInt8) {
                    zp_val = static_cast<int64_t>(Data[offset++]); // [0, 255]
                } else {
                    zp_val = static_cast<int64_t>(Data[offset++]) - 128; // [-128, 127]
                }
            }
            zero_points[i] = zp_val;
        }
        
        // Call quantize_per_channel
        torch::Tensor quantized = torch::quantize_per_channel(
            input_tensor,
            scales,
            zero_points,
            axis,
            dtype
        );
        
        // Test dequantization to exercise more code paths
        torch::Tensor dequantized = quantized.dequantize();
        
        // Verify the result has expected properties
        (void)quantized.q_per_channel_scales();
        (void)quantized.q_per_channel_zero_points();
        (void)quantized.q_per_channel_axis();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}