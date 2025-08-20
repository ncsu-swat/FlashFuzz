#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse quantization parameters
        // We need scale, zero_point, and axis
        
        // Create scale tensor (per-channel)
        torch::Tensor scales;
        if (offset < Size) {
            scales = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure scales are positive (required by fake_quantize_per_channel_affine)
            scales = torch::abs(scales) + 1e-10;
        } else {
            // Default scale if we don't have enough data
            scales = torch::ones({1});
        }
        
        // Create zero_points tensor (per-channel)
        torch::Tensor zero_points;
        if (offset < Size) {
            zero_points = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert to int64 as required by the API
            zero_points = zero_points.to(torch::kInt64);
        } else {
            // Default zero_points if we don't have enough data
            zero_points = torch::zeros({1}, torch::kInt64);
        }
        
        // Parse axis
        int64_t axis = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&axis, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure axis is within valid range for the input tensor
            if (input_tensor.dim() > 0) {
                axis = axis % input_tensor.dim();
                if (axis < 0) {
                    axis += input_tensor.dim();
                }
            } else {
                axis = 0;
            }
        }
        
        // Parse quant_min and quant_max
        int64_t quant_min = 0;
        int64_t quant_max = 255;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&quant_min, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&quant_max, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure quant_min < quant_max
        if (quant_min >= quant_max) {
            std::swap(quant_min, quant_max);
            if (quant_min == quant_max) {
                quant_min--;
            }
        }
        
        // Ensure input tensor has float-like dtype
        if (input_tensor.dtype() != torch::kFloat && 
            input_tensor.dtype() != torch::kDouble && 
            input_tensor.dtype() != torch::kHalf && 
            input_tensor.dtype() != torch::kBFloat16) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Ensure scales and zero_points have compatible sizes with the input tensor
        if (input_tensor.dim() > 0) {
            int64_t expected_size = input_tensor.size(axis);
            
            if (scales.numel() != expected_size) {
                scales = scales.expand({expected_size});
            }
            
            if (zero_points.numel() != expected_size) {
                zero_points = zero_points.expand({expected_size});
            }
        }
        
        // Apply fake_quantize_per_channel_affine
        torch::Tensor output = torch::fake_quantize_per_channel_affine(
            input_tensor, scales, zero_points, axis, quant_min, quant_max);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Prevent compiler from optimizing away the computation
        if (sum.item<float>() == -12345.6789f) {
            std::cerr << "Unlikely value encountered" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}