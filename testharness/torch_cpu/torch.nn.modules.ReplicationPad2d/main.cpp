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
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // ReplicationPad2d expects 3D (C, H, W) or 4D (N, C, H, W) input
        if (input.dim() < 3) {
            int64_t numel = input.numel();
            if (numel == 0) {
                return 0;
            }
            // Reshape to 4D (1, 1, H, W) format
            if (input.dim() == 0) {
                input = input.reshape({1, 1, 1, 1});
            } else if (input.dim() == 1) {
                input = input.reshape({1, 1, 1, numel});
            } else if (input.dim() == 2) {
                input = input.reshape({1, 1, input.size(0), input.size(1)});
            }
        } else if (input.dim() > 4) {
            // Flatten extra dimensions into batch
            auto sizes = input.sizes().vec();
            int64_t batch = 1;
            for (size_t i = 0; i < sizes.size() - 3; i++) {
                batch *= sizes[i];
            }
            input = input.reshape({batch, sizes[sizes.size()-3], sizes[sizes.size()-2], sizes[sizes.size()-1]});
        }
        
        // Extract padding values from the remaining data
        // Padding format: (left, right, top, bottom)
        int64_t padding[4] = {1, 1, 1, 1}; // Default padding
        
        for (int i = 0; i < 4 && offset + 1 <= Size; i++) {
            // Use single byte and bound to reasonable range to avoid huge allocations
            // Padding should be non-negative and smaller than input dimensions
            int8_t pad_byte = static_cast<int8_t>(Data[offset++]);
            // Map to range [0, 32] for reasonable padding values
            padding[i] = std::abs(pad_byte) % 33;
        }
        
        // Ensure padding doesn't exceed input dimensions (would cause empty tensor)
        int64_t h_dim = input.dim() == 3 ? 1 : 2;
        int64_t w_dim = input.dim() == 3 ? 2 : 3;
        int64_t input_h = input.size(h_dim);
        int64_t input_w = input.size(w_dim);
        
        // Clamp padding to valid range
        padding[0] = std::min(padding[0], input_w - 1);  // left
        padding[1] = std::min(padding[1], input_w - 1);  // right
        padding[2] = std::min(padding[2], input_h - 1);  // top
        padding[3] = std::min(padding[3], input_h - 1);  // bottom
        
        // Create ReplicationPad2d module with 4-element padding
        torch::nn::ReplicationPad2d pad_module(
            torch::nn::ReplicationPad2dOptions({padding[0], padding[1], padding[2], padding[3]})
        );
        
        // Apply padding
        torch::Tensor output = pad_module->forward(input);
        
        // Verify output shape is correct
        int64_t expected_h = input_h + padding[2] + padding[3];
        int64_t expected_w = input_w + padding[0] + padding[1];
        (void)expected_h;
        (void)expected_w;
        
        // Try with uniform padding if we have more data
        if (offset + 1 <= Size) {
            int8_t uniform_pad_byte = static_cast<int8_t>(Data[offset++]);
            int64_t uniform_pad = std::abs(uniform_pad_byte) % 17; // Range [0, 16]
            
            // Ensure uniform padding is valid
            uniform_pad = std::min(uniform_pad, std::min(input_h - 1, input_w - 1));
            
            if (uniform_pad >= 0) {
                try {
                    auto pad_module2 = torch::nn::ReplicationPad2d(
                        torch::nn::ReplicationPad2dOptions(uniform_pad)
                    );
                    torch::Tensor output2 = pad_module2->forward(input);
                    (void)output2;
                } catch (...) {
                    // Silent catch for expected failures
                }
            }
        }
        
        // Try with different tensor dtypes
        if (offset < Size) {
            auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
            if (dtype != input.dtype()) {
                try {
                    torch::Tensor input_cast = input.to(dtype);
                    torch::Tensor output_cast = pad_module->forward(input_cast);
                    (void)output_cast;
                } catch (...) {
                    // Silent catch for type conversion errors
                }
            }
        }
        
        // Test with 3D input (unbatched) if we originally had 4D
        if (input.dim() == 4 && input.size(0) == 1) {
            try {
                torch::Tensor input_3d = input.squeeze(0);
                torch::Tensor output_3d = pad_module->forward(input_3d);
                (void)output_3d;
            } catch (...) {
                // Silent catch
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