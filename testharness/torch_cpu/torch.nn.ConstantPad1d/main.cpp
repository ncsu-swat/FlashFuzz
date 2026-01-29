#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        // Extract padding values first
        int64_t padding_left = static_cast<int64_t>(Data[offset++] % 32);  // Limit padding size
        int64_t padding_right = static_cast<int64_t>(Data[offset++] % 32);
        
        // Get value to pad with
        float pad_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize pad_value to avoid NaN/Inf issues
            if (std::isnan(pad_value) || std::isinf(pad_value)) {
                pad_value = 0.0f;
            }
        }
        
        // Create input tensor - ConstantPad1d expects 2D (unbatched) or 3D (batched) input
        // Format: (N, C, W) or (C, W)
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape to appropriate dimensions for 1D padding
        int64_t numel = input_tensor.numel();
        if (numel < 1) {
            return 0;
        }
        
        // Determine tensor shape based on fuzzer byte
        uint8_t shape_selector = (offset < Size) ? Data[offset++] % 3 : 0;
        
        try {
            if (shape_selector == 0 && numel >= 1) {
                // 2D: (C, W)
                input_tensor = input_tensor.reshape({1, numel});
            } else if (shape_selector == 1 && numel >= 2) {
                // 3D: (N, C, W) with N=1, C=1
                input_tensor = input_tensor.reshape({1, 1, numel});
            } else if (numel >= 4) {
                // 3D: (N, C, W) with derived dimensions
                int64_t w = std::max<int64_t>(1, numel / 4);
                int64_t c = std::max<int64_t>(1, numel / (w * 2));
                int64_t n = numel / (c * w);
                if (n * c * w != numel) {
                    input_tensor = input_tensor.reshape({1, 1, numel});
                } else {
                    input_tensor = input_tensor.reshape({n, c, w});
                }
            } else {
                input_tensor = input_tensor.reshape({1, numel});
            }
        } catch (...) {
            // Fallback to simple 2D shape
            input_tensor = input_tensor.flatten().reshape({1, -1});
        }
        
        // Create padding options - for 1D padding, vector is {left, right}
        std::vector<int64_t> padding = {padding_left, padding_right};
        
        // Test using torch::nn::functional::pad with constant mode
        torch::Tensor output = torch::nn::functional::pad(
            input_tensor,
            torch::nn::functional::PadFuncOptions(padding)
                .mode(torch::kConstant)
                .value(pad_value)
        );
        
        // Force computation
        output = output.contiguous();
        
        // Verify output shape is correct
        int64_t last_dim = input_tensor.size(-1);
        int64_t expected_last_dim = last_dim + padding_left + padding_right;
        if (output.size(-1) != expected_last_dim) {
            std::cerr << "Unexpected output dimension" << std::endl;
        }
        
        // Access elements to ensure computation is performed
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Also test the ConstantPad1d module directly
        try {
            torch::nn::ConstantPad1d pad_module(
                torch::nn::ConstantPad1dOptions({padding_left, padding_right}, pad_value)
            );
            torch::Tensor module_output = pad_module(input_tensor);
            module_output = module_output.contiguous();
            
            if (module_output.numel() > 0) {
                volatile float module_sum = module_output.sum().item<float>();
                (void)module_sum;
            }
        } catch (...) {
            // Module variant may have different constraints, ignore errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}