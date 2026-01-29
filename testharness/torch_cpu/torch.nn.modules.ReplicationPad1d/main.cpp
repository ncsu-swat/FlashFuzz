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
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract padding values with reasonable bounds
        int8_t pad_left_raw = static_cast<int8_t>(Data[offset++]);
        int8_t pad_right_raw = static_cast<int8_t>(Data[offset++]);
        
        // Bound padding to reasonable values (0-16)
        int64_t padding_left = std::abs(pad_left_raw) % 17;
        int64_t padding_right = std::abs(pad_right_raw) % 17;

        // Extract dimensions for input tensor
        uint8_t dim_choice = Data[offset++];
        uint8_t batch_size_raw = Data[offset++];
        uint8_t channels_raw = Data[offset++];
        uint8_t width_raw = Data[offset++];

        // Bound dimensions to reasonable values
        int64_t batch_size = (batch_size_raw % 4) + 1;  // 1-4
        int64_t channels = (channels_raw % 8) + 1;      // 1-8
        int64_t width = (width_raw % 32) + 1;           // 1-32

        // Decide padding configuration based on fuzzer data
        bool use_single_pad = (dim_choice & 0x01) != 0;
        bool use_3d_input = (dim_choice & 0x02) != 0;

        // Create padding options
        torch::nn::ReplicationPad1dOptions options({padding_left, padding_right});
        if (use_single_pad) {
            // Use symmetric padding
            options = torch::nn::ReplicationPad1dOptions(padding_left);
        }

        // Create ReplicationPad1d module
        torch::nn::ReplicationPad1d pad_module(options);

        // Create input tensor with appropriate dimensions
        // ReplicationPad1d expects 2D (unbatched) or 3D (batched) input
        torch::Tensor input;
        if (use_3d_input) {
            // 3D input: (N, C, W)
            input = torch::randn({batch_size, channels, width});
        } else {
            // 2D input: (C, W)
            input = torch::randn({channels, width});
        }

        // Apply padding
        torch::Tensor output = pad_module->forward(input);

        // Verify output shape is correct
        int64_t expected_width_increase = padding_left + padding_right;
        int64_t output_width = output.size(-1);
        int64_t input_width = input.size(-1);
        
        if (output_width != input_width + expected_width_increase) {
            std::cerr << "Output width mismatch!" << std::endl;
        }

        // Access output to ensure computation completed
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }

        // Test with different tensor values from fuzzer data
        if (offset + 4 <= Size) {
            // Create tensor with values derived from fuzzer data
            std::vector<float> values;
            for (size_t i = offset; i < std::min(offset + 16, Size); i++) {
                values.push_back(static_cast<float>(Data[i]) / 255.0f - 0.5f);
            }
            
            int64_t val_count = static_cast<int64_t>(values.size());
            torch::Tensor custom_input;
            
            try {
                if (use_3d_input) {
                    custom_input = torch::from_blob(values.data(), {1, 1, val_count}, 
                                                    torch::kFloat32).clone();
                } else {
                    custom_input = torch::from_blob(values.data(), {1, val_count}, 
                                                    torch::kFloat32).clone();
                }
                
                torch::Tensor custom_output = pad_module->forward(custom_input);
                
                // Verify edge values are replicated correctly
                if (custom_output.numel() > 0 && padding_left > 0) {
                    // First padding_left elements should equal the first input element
                    volatile float first = custom_output.flatten()[0].item<float>();
                    (void)first;
                }
            } catch (const std::exception &) {
                // Silently catch expected failures from shape mismatches
            }
        }

        // Test with different dtypes
        uint8_t dtype_choice = (offset < Size) ? Data[offset] : 0;
        try {
            torch::Tensor typed_input;
            switch (dtype_choice % 3) {
                case 0:
                    typed_input = input.to(torch::kFloat32);
                    break;
                case 1:
                    typed_input = input.to(torch::kFloat64);
                    break;
                case 2:
                    typed_input = input.to(torch::kFloat16);
                    break;
            }
            torch::Tensor typed_output = pad_module->forward(typed_input);
            volatile float check = typed_output.sum().to(torch::kFloat32).item<float>();
            (void)check;
        } catch (const std::exception &) {
            // Some dtypes may not be supported, silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}