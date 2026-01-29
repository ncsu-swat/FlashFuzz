#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        // Need enough bytes for configuration
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract dimensions for 5D tensor (N, C, D, H, W)
        // Use small dimensions to avoid memory issues
        int64_t batch = (Data[offset++] % 4) + 1;      // 1-4
        int64_t channels = (Data[offset++] % 4) + 1;   // 1-4
        int64_t depth = (Data[offset++] % 8) + 1;      // 1-8
        int64_t height = (Data[offset++] % 8) + 1;     // 1-8
        int64_t width = (Data[offset++] % 8) + 1;      // 1-8
        
        // Extract padding values (use non-negative values to avoid issues)
        // ConstantPad3d expects padding for (left, right, top, bottom, front, back)
        int64_t pad_left = Data[offset++] % 5;
        int64_t pad_right = Data[offset++] % 5;
        int64_t pad_top = Data[offset++] % 5;
        int64_t pad_bottom = Data[offset++] % 5;
        int64_t pad_front = Data[offset++] % 5;
        int64_t pad_back = Data[offset++] % 5;
        
        // Get value to pad with
        double pad_value = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize pad_value to avoid NaN/Inf issues
            if (std::isnan(pad_value) || std::isinf(pad_value)) {
                pad_value = 0.0;
            }
        }
        
        // Determine dtype from remaining data
        int dtype_selector = (offset < Size) ? Data[offset++] % 3 : 0;
        torch::Dtype dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create 5D input tensor for ConstantPad3d
        torch::Tensor input_tensor = torch::randn({batch, channels, depth, height, width}, 
                                                   torch::TensorOptions().dtype(dtype));
        
        // Create the ConstantPad3d module
        torch::nn::ConstantPad3d pad_module(
            torch::nn::ConstantPad3dOptions(
                {pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back}, 
                pad_value
            )
        );
        
        // Apply padding
        torch::Tensor output = pad_module->forward(input_tensor);
        
        // Verify output
        if (output.defined() && output.numel() > 0) {
            auto sum = output.sum();
            (void)sum; // Ensure computation happens
        }
        
        // Also test with 4D input (C, D, H, W) - unbatched case
        torch::Tensor input_4d = torch::randn({channels, depth, height, width},
                                               torch::TensorOptions().dtype(dtype));
        torch::Tensor output_4d = pad_module->forward(input_4d);
        
        if (output_4d.defined() && output_4d.numel() > 0) {
            auto mean = output_4d.mean();
            (void)mean;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}