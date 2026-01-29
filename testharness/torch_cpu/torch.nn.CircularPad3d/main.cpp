#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 5D tensor (batch, channels, depth, height, width)
        // CircularPad3d requires 4D or 5D input
        int64_t numel = input_tensor.numel();
        
        if (numel == 0) {
            return 0;
        }
        
        // Flatten and reshape to 5D
        input_tensor = input_tensor.flatten();
        
        // Create reasonable dimensions
        int64_t batch = 1;
        int64_t channels = 1;
        int64_t depth = std::max(int64_t(1), std::min(int64_t(8), numel));
        int64_t height = std::max(int64_t(1), std::min(int64_t(8), numel / depth));
        int64_t width = std::max(int64_t(1), numel / (depth * height));
        
        int64_t needed = batch * channels * depth * height * width;
        if (needed > numel) {
            // Adjust to fit
            width = numel / (batch * channels * depth * height);
            if (width < 1) width = 1;
            needed = batch * channels * depth * height * width;
        }
        
        input_tensor = input_tensor.narrow(0, 0, needed).reshape({batch, channels, depth, height, width});
        
        // Parse padding values from remaining data - bound them to reasonable values
        int64_t padding[6] = {1, 1, 1, 1, 1, 1};
        for (int i = 0; i < 6 && offset < Size; i++) {
            // Use single byte for padding, bounded to tensor dimensions
            int64_t max_pad;
            if (i < 2) {
                max_pad = width - 1;  // width padding
            } else if (i < 4) {
                max_pad = height - 1; // height padding
            } else {
                max_pad = depth - 1;  // depth padding
            }
            max_pad = std::max(int64_t(0), max_pad);
            
            padding[i] = Data[offset++] % (max_pad + 1);
        }
        
        // Get configuration byte
        uint8_t config = 0;
        if (offset < Size) {
            config = Data[offset++];
        }
        
        torch::Tensor output;
        
        // torch::nn::CircularPad3d module doesn't exist in C++ frontend
        // Use functional API with circular mode instead
        // The functional pad with kCircular mode provides equivalent functionality
        
        if (config % 3 == 0) {
            // Test with 6-value padding (left, right, top, bottom, front, back)
            try {
                output = torch::nn::functional::pad(input_tensor,
                    torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]})
                    .mode(torch::kCircular));
            } catch (const c10::Error&) {
                // Expected for invalid configurations
                return 0;
            }
        } else if (config % 3 == 1) {
            // Test with 4-value padding (left, right, top, bottom)
            try {
                output = torch::nn::functional::pad(input_tensor,
                    torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], padding[3]})
                    .mode(torch::kCircular));
            } catch (const c10::Error&) {
                // Expected for invalid configurations
                return 0;
            }
        } else {
            // Test with 2-value padding (left, right)
            try {
                output = torch::nn::functional::pad(input_tensor,
                    torch::nn::functional::PadFuncOptions({padding[0], padding[1]})
                    .mode(torch::kCircular));
            } catch (const c10::Error&) {
                // Expected for invalid configurations
                return 0;
            }
        }
        
        // Force computation
        output.sum().item<float>();
        
        // Test with 4D input (unbatched: channels, depth, height, width)
        if (config % 8 >= 4 && input_tensor.dim() == 5) {
            try {
                auto input_4d = input_tensor.squeeze(0);
                auto output_4d = torch::nn::functional::pad(input_4d,
                    torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]})
                    .mode(torch::kCircular));
                output_4d.sum().item<float>();
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }
        
        // Test with different tensor dtypes
        if (config % 16 >= 8) {
            try {
                auto input_double = input_tensor.to(torch::kDouble);
                auto output_double = torch::nn::functional::pad(input_double,
                    torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]})
                    .mode(torch::kCircular));
                output_double.sum().item<double>();
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}