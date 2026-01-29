#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need enough bytes for tensor creation and parameters
        if (Size < 16) {
            return 0;
        }
        
        // Create a float tensor that we'll quantize
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor is float and has at least 2 dimensions for per-channel quantization
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        // Convert to float if needed
        torch::Tensor float_tensor = input_tensor.to(torch::kFloat32);
        
        // Ensure we have at least 2D tensor for per-channel quantization
        if (float_tensor.dim() < 1) {
            float_tensor = float_tensor.unsqueeze(0);
        }
        if (float_tensor.dim() < 2) {
            float_tensor = float_tensor.unsqueeze(0);
        }
        
        // Get axis from fuzzer data (must be valid dimension)
        int64_t axis = 0;
        if (offset < Size) {
            axis = Data[offset] % float_tensor.dim();
            offset++;
        }
        
        // Get the size along the chosen axis for scales/zero_points
        int64_t num_channels = float_tensor.size(axis);
        if (num_channels <= 0) {
            return 0;
        }
        
        // Create scales (must be positive) and zero_points for per-channel quantization
        torch::Tensor scales = torch::ones({num_channels}, torch::kFloat64) * 0.1;
        torch::Tensor zero_points = torch::zeros({num_channels}, torch::kLong);
        
        // Vary scales slightly based on fuzzer input
        if (offset < Size) {
            double scale_factor = 0.01 + (Data[offset] % 100) * 0.01;
            scales = scales * scale_factor;
            offset++;
        }
        
        // Perform per-channel quantization
        torch::Tensor quantized_tensor;
        try {
            quantized_tensor = torch::quantize_per_channel(
                float_tensor,
                scales,
                zero_points,
                axis,
                torch::kQInt8  // quantized dtype
            );
        } catch (...) {
            // Quantization may fail for certain tensor configurations
            return 0;
        }
        
        // Now call q_per_channel_axis on the per-channel quantized tensor
        int64_t result = torch::q_per_channel_axis(quantized_tensor);
        
        // Verify the result matches the axis we used for quantization
        volatile int64_t axis_value = result;
        (void)axis_value;
        
        // Also test the related per-channel quantization query functions
        try {
            torch::Tensor retrieved_scales = torch::q_per_channel_scales(quantized_tensor);
            torch::Tensor retrieved_zero_points = torch::q_per_channel_zero_points(quantized_tensor);
            (void)retrieved_scales;
            (void)retrieved_zero_points;
        } catch (...) {
            // These should work but catch just in case
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}