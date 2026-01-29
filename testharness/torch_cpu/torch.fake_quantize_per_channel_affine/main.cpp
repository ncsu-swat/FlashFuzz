#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Early exit if not enough data
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip 0-dim tensors as fake_quantize_per_channel_affine requires at least 1D
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Ensure input tensor has float dtype (required by the API)
        if (input_tensor.dtype() != torch::kFloat && 
            input_tensor.dtype() != torch::kDouble && 
            input_tensor.dtype() != torch::kHalf && 
            input_tensor.dtype() != torch::kBFloat16) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Parse axis first so we know the expected size for scales and zero_points
        int64_t axis = 0;
        if (offset + sizeof(uint8_t) <= Size) {
            axis = static_cast<int64_t>(Data[offset]) % input_tensor.dim();
            offset += sizeof(uint8_t);
        }
        
        int64_t channel_size = input_tensor.size(axis);
        
        // Skip if channel size is 0
        if (channel_size == 0) {
            return 0;
        }
        
        // Create scale tensor (per-channel) - must be 1D float tensor
        torch::Tensor scales;
        if (offset < Size) {
            scales = fuzzer_utils::createTensor(Data, Size, offset);
            // Flatten to 1D
            scales = scales.flatten();
            // Convert to float
            scales = scales.to(torch::kFloat);
            // Ensure scales are positive (required by fake_quantize_per_channel_affine)
            scales = torch::abs(scales) + 1e-6f;
        } else {
            scales = torch::ones({channel_size}, torch::kFloat);
        }
        
        // Resize scales to match channel_size
        if (scales.numel() < channel_size) {
            // Repeat to fill
            int64_t repeat_count = (channel_size + scales.numel() - 1) / scales.numel();
            scales = scales.repeat({repeat_count}).slice(0, 0, channel_size);
        } else if (scales.numel() > channel_size) {
            scales = scales.slice(0, 0, channel_size);
        }
        scales = scales.contiguous();
        
        // Create zero_points tensor (per-channel) - must be 1D int tensor
        torch::Tensor zero_points;
        if (offset < Size) {
            zero_points = fuzzer_utils::createTensor(Data, Size, offset);
            // Flatten to 1D
            zero_points = zero_points.flatten();
        } else {
            zero_points = torch::zeros({channel_size});
        }
        
        // Resize zero_points to match channel_size
        if (zero_points.numel() < channel_size) {
            int64_t repeat_count = (channel_size + zero_points.numel() - 1) / zero_points.numel();
            zero_points = zero_points.repeat({repeat_count}).slice(0, 0, channel_size);
        } else if (zero_points.numel() > channel_size) {
            zero_points = zero_points.slice(0, 0, channel_size);
        }
        // Convert to int64 as required by the API
        zero_points = zero_points.to(torch::kInt64).contiguous();
        
        // Parse quant_min and quant_max
        int64_t quant_min = 0;
        int64_t quant_max = 255;
        
        if (offset + sizeof(uint8_t) <= Size) {
            quant_min = static_cast<int64_t>(Data[offset]);
            offset += sizeof(uint8_t);
        }
        
        if (offset + sizeof(uint8_t) <= Size) {
            quant_max = static_cast<int64_t>(Data[offset]);
            offset += sizeof(uint8_t);
        }
        
        // Ensure quant_min < quant_max
        if (quant_min >= quant_max) {
            std::swap(quant_min, quant_max);
        }
        if (quant_min == quant_max) {
            quant_max = quant_min + 1;
        }
        
        // Clamp zero_points to valid range
        zero_points = torch::clamp(zero_points, quant_min, quant_max);
        
        // Apply fake_quantize_per_channel_affine
        try {
            torch::Tensor output = torch::fake_quantize_per_channel_affine(
                input_tensor, scales, zero_points, axis, quant_min, quant_max);
            
            // Perform some operations on the output to ensure it's used
            auto sum = output.sum();
            
            // Prevent compiler from optimizing away the computation
            if (sum.item<float>() == -12345.6789f) {
                std::cerr << "Unlikely value encountered" << std::endl;
            }
        }
        catch (const c10::Error &e) {
            // Silently catch expected errors like shape mismatches
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}