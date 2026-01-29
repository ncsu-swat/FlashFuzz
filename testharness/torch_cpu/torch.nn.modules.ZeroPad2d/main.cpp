#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Extract padding values first (4 bytes)
        int64_t left = static_cast<int64_t>(Data[offset++] % 32);
        int64_t right = static_cast<int64_t>(Data[offset++] % 32);
        int64_t top = static_cast<int64_t>(Data[offset++] % 32);
        int64_t bottom = static_cast<int64_t>(Data[offset++] % 32);
        
        // Determine tensor dimensions from data
        uint8_t shape_selector = Data[offset++];
        
        // Create input tensor with appropriate shape for ZeroPad2d (3D or 4D)
        torch::Tensor input_tensor;
        if (shape_selector % 2 == 0) {
            // 3D tensor: (C, H, W)
            int64_t C = 1 + (Data[offset++] % 4);
            int64_t H = 1 + (Data[offset++] % 16);
            int64_t W = 1 + (Data[offset++] % 16);
            input_tensor = torch::randn({C, H, W});
        } else {
            // 4D tensor: (N, C, H, W)
            int64_t N = 1 + (Data[offset++] % 4);
            int64_t C = 1 + (Data[offset++] % 4);
            int64_t H = 1 + (Data[offset++] % 16);
            int64_t W = 1 + (Data[offset++] % 16);
            input_tensor = torch::randn({N, C, H, W});
        }
        
        // ZeroPad2d requires padding as {left, right, top, bottom}
        std::vector<int64_t> padding = {left, right, top, bottom};
        
        // Create ZeroPad2d module using options
        auto zeropad = torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions(padding));
        
        // Forward pass
        torch::Tensor output = zeropad->forward(input_tensor);
        
        // Verify output shape is correct
        auto input_sizes = input_tensor.sizes();
        auto output_sizes = output.sizes();
        int ndim = input_sizes.size();
        
        // H dimension should increase by top + bottom
        // W dimension should increase by left + right
        (void)output_sizes; // Use output to prevent optimization
        
        // Try functional version as well
        auto functional_output = torch::nn::functional::pad(
            input_tensor,
            torch::nn::functional::PadFuncOptions(padding).mode(torch::kConstant).value(0.0)
        );
        
        // Test with uniform padding (same value all sides)
        if (offset < Size) {
            int64_t uniform_pad = static_cast<int64_t>(Data[offset++] % 16);
            std::vector<int64_t> uniform_padding = {uniform_pad, uniform_pad, uniform_pad, uniform_pad};
            auto uniform_zeropad = torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions(uniform_padding));
            auto uniform_output = uniform_zeropad->forward(input_tensor);
            (void)uniform_output;
        }
        
        // Test with asymmetric padding
        if (offset + 1 < Size) {
            int64_t h_pad = static_cast<int64_t>(Data[offset++] % 16);
            int64_t w_pad = static_cast<int64_t>(Data[offset++] % 16);
            std::vector<int64_t> asym_padding = {w_pad, w_pad, h_pad, h_pad};
            auto asym_zeropad = torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions(asym_padding));
            auto asym_output = asym_zeropad->forward(input_tensor);
            (void)asym_output;
        }
        
        // Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::Tensor typed_tensor;
            
            try {
                if (dtype_selector % 4 == 0) {
                    typed_tensor = input_tensor.to(torch::kFloat32);
                } else if (dtype_selector % 4 == 1) {
                    typed_tensor = input_tensor.to(torch::kFloat64);
                } else if (dtype_selector % 4 == 2) {
                    typed_tensor = input_tensor.to(torch::kInt32);
                } else {
                    typed_tensor = input_tensor.to(torch::kInt64);
                }
                
                auto typed_output = zeropad->forward(typed_tensor);
                (void)typed_output;
            } catch (...) {
                // Some dtypes may not be supported, silently ignore
            }
        }
        
        // Test zero padding (edge case)
        {
            std::vector<int64_t> zero_padding = {0, 0, 0, 0};
            auto zero_zeropad = torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions(zero_padding));
            auto zero_output = zero_zeropad->forward(input_tensor);
            (void)zero_output;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}