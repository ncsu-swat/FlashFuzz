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
        
        // Need enough bytes for tensor and padding values
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract padding values from the remaining data (6 values for 3D: left, right, top, bottom, front, back)
        std::array<int64_t, 6> padding;
        for (int i = 0; i < 6; ++i) {
            if (offset < Size) {
                // Use single byte and constrain to reasonable padding values [0, 16]
                padding[i] = static_cast<int64_t>(Data[offset] % 17);
                offset++;
            } else {
                padding[i] = 1;
            }
        }
        
        // Ensure input has at least 4D for ReplicationPad3d (can be 4D or 5D)
        // 4D: (N, C, D, H, W) without batch or (C, D, H, W)
        // 5D: (N, C, D, H, W)
        if (input.dim() < 4) {
            std::vector<int64_t> new_shape;
            for (int64_t i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            while (new_shape.size() < 5) {
                new_shape.insert(new_shape.begin(), 1);
            }
            input = input.reshape(new_shape);
        } else if (input.dim() > 5) {
            // Flatten extra dimensions into batch
            input = input.reshape({-1, input.size(-4), input.size(-3), input.size(-2), input.size(-1)});
        }
        
        // Ensure minimum size in spatial dimensions for replication padding
        // Each spatial dim must be > 0
        bool valid_dims = true;
        for (int64_t i = input.dim() - 3; i < input.dim(); i++) {
            if (input.size(i) < 1) {
                valid_dims = false;
                break;
            }
        }
        if (!valid_dims) {
            return 0;
        }
        
        // Create ReplicationPad3d module with array of 6 padding values
        torch::nn::ReplicationPad3d pad_module(padding);
        
        // Apply padding
        torch::Tensor output = pad_module->forward(input);
        
        // Verify output shape is reasonable
        if (output.numel() == 0) {
            return 0;
        }
        
        // Test with single padding value
        if (offset < Size) {
            int64_t single_pad = static_cast<int64_t>(Data[offset] % 9);
            offset++;
            
            try {
                torch::nn::ReplicationPad3d pad_module2(single_pad);
                torch::Tensor output2 = pad_module2->forward(input);
            } catch (const std::exception&) {
                // Silently handle expected failures
            }
        }
        
        // Try with different data types
        try {
            if (input.dtype() != torch::kFloat) {
                auto input_float = input.to(torch::kFloat);
                torch::Tensor output_float = pad_module->forward(input_float);
            }
            
            if (input.dtype() != torch::kDouble) {
                auto input_double = input.to(torch::kDouble);
                torch::Tensor output_double = pad_module->forward(input_double);
            }
        } catch (const std::exception&) {
            // Silently handle type conversion issues
        }
        
        // Test with asymmetric padding
        if (offset + 5 < Size) {
            std::array<int64_t, 6> asym_padding;
            for (int i = 0; i < 6; ++i) {
                asym_padding[i] = static_cast<int64_t>(Data[offset + i] % 9);
            }
            offset += 6;
            
            try {
                torch::nn::ReplicationPad3d asym_module(asym_padding);
                torch::Tensor asym_output = asym_module->forward(input);
            } catch (const std::exception&) {
                // Silently handle failures
            }
        }
        
        // Test with zero padding (should be a no-op)
        try {
            std::array<int64_t, 6> zero_padding = {0, 0, 0, 0, 0, 0};
            torch::nn::ReplicationPad3d zero_module(zero_padding);
            torch::Tensor zero_output = zero_module->forward(input);
        } catch (const std::exception&) {
            // Silently handle failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}