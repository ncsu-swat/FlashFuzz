#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least a few bytes for meaningful fuzzing
        if (Size < 8) {
            return 0;
        }
        
        // Extract padding parameters first
        int64_t pad_left = static_cast<int64_t>(Data[offset++] % 64);
        int64_t pad_right = static_cast<int64_t>(Data[offset++] % 64);
        
        // Extract value to pad with
        float pad_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize pad_value to avoid NaN/Inf issues
            if (std::isnan(pad_value) || std::isinf(pad_value)) {
                pad_value = 0.0f;
            }
        }
        
        // Create input tensor - ConstantPad1d expects 2D or 3D input
        // For 2D: (N, W) or for 3D: (N, C, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has appropriate dimensions for ConstantPad1d
        if (input.dim() < 2) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 2) {
            input = input.unsqueeze(0);
        }
        // Limit to 3D for ConstantPad1d
        while (input.dim() > 3) {
            input = input.squeeze(0);
        }
        
        // Create the ConstantPad1d module with asymmetric padding
        torch::nn::ConstantPad1d pad(torch::nn::ConstantPad1dOptions({pad_left, pad_right}, pad_value));
        
        // Apply padding
        torch::Tensor output = pad(input);
        
        // Try symmetric padding configuration
        if (offset < Size) {
            int64_t sym_pad = static_cast<int64_t>(Data[offset++] % 32);
            try {
                torch::nn::ConstantPad1d sym_pad_module(torch::nn::ConstantPad1dOptions(sym_pad, pad_value));
                torch::Tensor sym_output = sym_pad_module(input);
            } catch (...) {
                // Silently ignore expected errors
            }
        }
        
        // Try with different tensor dtypes
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::Tensor double_output = pad(double_input);
        } catch (...) {
            // Silently ignore conversion errors
        }
        
        // Try with zero padding
        try {
            torch::nn::ConstantPad1d zero_pad(torch::nn::ConstantPad1dOptions({0, 0}, 0.0));
            torch::Tensor zero_output = zero_pad(input);
        } catch (...) {
            // Silently ignore errors
        }
        
        // Try negative padding (cropping) if tensor is large enough
        if (offset + 2 <= Size && input.size(-1) > 4) {
            int64_t neg_left = -(static_cast<int64_t>(Data[offset++] % 2) + 1);
            int64_t neg_right = -(static_cast<int64_t>(Data[offset++] % 2) + 1);
            
            // Ensure we don't crop more than the tensor size
            int64_t max_crop = input.size(-1) / 2;
            neg_left = std::max(neg_left, -max_crop);
            neg_right = std::max(neg_right, -max_crop);
            
            try {
                torch::nn::ConstantPad1d neg_pad(torch::nn::ConstantPad1dOptions({neg_left, neg_right}, pad_value));
                torch::Tensor neg_output = neg_pad(input);
            } catch (...) {
                // Silently ignore errors from negative padding
            }
        }
        
        // Try with a different constant value
        if (offset + sizeof(float) <= Size) {
            float alt_value;
            std::memcpy(&alt_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (!std::isnan(alt_value) && !std::isinf(alt_value)) {
                try {
                    torch::nn::ConstantPad1d alt_pad(torch::nn::ConstantPad1dOptions({pad_left, pad_right}, alt_value));
                    torch::Tensor alt_output = alt_pad(input);
                } catch (...) {
                    // Silently ignore errors
                }
            }
        }
        
        // Try with 3D input if we have 2D
        if (input.dim() == 2) {
            try {
                torch::Tensor input_3d = input.unsqueeze(1);
                torch::Tensor output_3d = pad(input_3d);
            } catch (...) {
                // Silently ignore errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}