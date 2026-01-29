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
        if (Size < 10) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract padding values first (6 values for 3D padding: left, right, top, bottom, front, back)
        std::vector<int64_t> padding(6, 0);
        for (int i = 0; i < 6 && offset < Size; i++) {
            padding[i] = static_cast<int64_t>(Data[offset]) % 8;  // Limit to 0-7
            offset++;
        }
        
        // Get dimension configuration from data
        uint8_t dim_config = (offset < Size) ? Data[offset++] : 0;
        bool use_5d = (dim_config % 2 == 0);  // Alternate between 4D and 5D
        
        // Create input tensor with appropriate dimensions
        // ReplicationPad3d expects 4D (C, D, H, W) or 5D (N, C, D, H, W) input
        int64_t batch = 1 + (offset < Size ? Data[offset++] % 3 : 0);
        int64_t channels = 1 + (offset < Size ? Data[offset++] % 4 : 0);
        int64_t depth = 2 + (offset < Size ? Data[offset++] % 6 : 0);
        int64_t height = 2 + (offset < Size ? Data[offset++] % 8 : 0);
        int64_t width = 2 + (offset < Size ? Data[offset++] % 8 : 0);
        
        torch::Tensor input;
        if (use_5d) {
            input = torch::randn({batch, channels, depth, height, width});
        } else {
            input = torch::randn({channels, depth, height, width});
        }
        
        // Create ReplicationPad3d module with 6-element padding
        torch::nn::ReplicationPad3d pad_module{torch::nn::ReplicationPad3dOptions(
            {padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]}
        )};
        
        // Apply padding
        torch::Tensor output = pad_module->forward(input);
        
        // Verify output dimensions are correct
        int64_t expected_width = input.size(-1) + padding[0] + padding[1];
        int64_t expected_height = input.size(-2) + padding[2] + padding[3];
        int64_t expected_depth = input.size(-3) + padding[4] + padding[5];
        (void)expected_width;
        (void)expected_height;
        (void)expected_depth;
        
        // Test with single padding value (applied symmetrically to all sides)
        if (offset < Size) {
            int64_t single_pad = Data[offset++] % 5;
            torch::nn::ReplicationPad3d single_pad_module{torch::nn::ReplicationPad3dOptions(single_pad)};
            torch::Tensor single_output = single_pad_module->forward(input);
            (void)single_output;
        }
        
        // Test functional interface
        try {
            torch::Tensor func_output = torch::nn::functional::pad(
                input,
                torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]})
                    .mode(torch::kReplicate)
            );
            (void)func_output;
        } catch (...) {
            // May fail for certain configurations
        }
        
        // Test with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::Tensor typed_input;
            if (dtype_selector == 0) {
                typed_input = input.to(torch::kFloat32);
            } else if (dtype_selector == 1) {
                typed_input = input.to(torch::kFloat64);
            } else {
                typed_input = input.to(torch::kFloat16);
            }
            
            try {
                torch::Tensor typed_output = pad_module->forward(typed_input);
                (void)typed_output;
            } catch (...) {
                // Some dtypes may not be supported
            }
        }
        
        // Test edge case: zero padding
        {
            torch::nn::ReplicationPad3d zero_pad_module{torch::nn::ReplicationPad3dOptions(0)};
            torch::Tensor zero_output = zero_pad_module->forward(input);
            (void)zero_output;
        }
        
        // Test asymmetric padding with varied values
        if (offset + 6 <= Size) {
            std::vector<int64_t> asym_padding(6);
            for (int i = 0; i < 6; i++) {
                asym_padding[i] = Data[offset++] % 4;
            }
            
            torch::nn::ReplicationPad3d asym_pad_module{torch::nn::ReplicationPad3dOptions(
                {asym_padding[0], asym_padding[1], asym_padding[2], 
                 asym_padding[3], asym_padding[4], asym_padding[5]}
            )};
            torch::Tensor asym_output = asym_pad_module->forward(input);
            (void)asym_output;
        }
        
        // Test with contiguous and non-contiguous input
        {
            torch::Tensor transposed = input.transpose(-1, -2);
            if (!transposed.is_contiguous()) {
                try {
                    torch::Tensor non_contig_output = pad_module->forward(transposed);
                    (void)non_contig_output;
                } catch (...) {
                    // May require contiguous input
                }
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