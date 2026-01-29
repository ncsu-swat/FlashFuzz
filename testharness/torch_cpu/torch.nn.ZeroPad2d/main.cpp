#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract padding values from data first (limit to reasonable values)
        int64_t left = static_cast<int64_t>(Data[offset++] % 32);
        int64_t right = static_cast<int64_t>(Data[offset++] % 32);
        int64_t top = static_cast<int64_t>(Data[offset++] % 32);
        int64_t bottom = static_cast<int64_t>(Data[offset++] % 32);
        
        // Create input tensor from remaining data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // ZeroPad2d expects 3D (C, H, W) or 4D (N, C, H, W) input
        // Reshape tensor to be 4D if needed
        int64_t numel = input_tensor.numel();
        if (numel < 1) {
            return 0;
        }
        
        // Create a 4D tensor with reasonable dimensions
        int64_t batch = 1;
        int64_t channels = 1;
        int64_t height = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(numel / 2)));
        int64_t width = numel / height;
        
        if (height * width < 1) {
            return 0;
        }
        
        // Flatten and take only what we need
        input_tensor = input_tensor.flatten().slice(0, 0, height * width);
        input_tensor = input_tensor.reshape({batch, channels, height, width});
        
        // Create ZeroPad2d module with proper options
        // ZeroPad2dOptions takes a tuple/vector of 4 ints: {left, right, top, bottom}
        torch::nn::ZeroPad2d zero_pad(torch::nn::ZeroPad2dOptions({left, right, top, bottom}));
        
        // Apply padding to the input tensor
        torch::Tensor output_tensor;
        try {
            output_tensor = zero_pad->forward(input_tensor);
        } catch (const c10::Error&) {
            // Shape mismatch or other expected errors
            return 0;
        }
        
        // Verify output dimensions are correct
        // Output should be: (N, C, H + top + bottom, W + left + right)
        auto out_sizes = output_tensor.sizes();
        if (out_sizes.size() != 4) {
            return 0;
        }
        
        // Test with 3D input as well (C, H, W)
        torch::Tensor input_3d = input_tensor.squeeze(0); // Remove batch dimension
        try {
            torch::Tensor output_3d = zero_pad->forward(input_3d);
            output_3d.sum().item<float>();
        } catch (const c10::Error&) {
            // Expected for some inputs
        }
        
        // Also test functional interface
        try {
            torch::Tensor output_func = torch::nn::functional::pad(
                input_tensor,
                torch::nn::functional::PadFuncOptions({left, right, top, bottom})
                    .mode(torch::kConstant)
                    .value(0)
            );
            output_func.sum().item<float>();
        } catch (const c10::Error&) {
            // Expected for some inputs
        }
        
        // Force computation
        output_tensor.sum().item<float>();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}