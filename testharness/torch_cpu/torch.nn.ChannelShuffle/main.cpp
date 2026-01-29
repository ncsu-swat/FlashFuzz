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
        
        // Need at least 4 bytes for parameters
        if (Size < 4) {
            return 0;
        }
        
        // Extract groups parameter (1-16, must be positive)
        int64_t groups = static_cast<int64_t>(Data[offset++] % 16) + 1;
        
        // Extract dimensions for 4D tensor (N, C, H, W)
        // ChannelShuffle requires 4D input with channels divisible by groups
        int64_t batch = static_cast<int64_t>(Data[offset++] % 4) + 1;
        int64_t height = static_cast<int64_t>(Data[offset++] % 8) + 1;
        int64_t width = static_cast<int64_t>(Data[offset++] % 8) + 1;
        
        // Ensure channels is divisible by groups
        int64_t channels = groups * (static_cast<int64_t>(offset < Size ? Data[offset++] % 8 : 1) + 1);
        
        // Determine call type
        uint8_t call_type = (offset < Size) ? Data[offset++] % 3 : 0;
        
        // Determine dtype
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
                case 3: dtype = torch::kBFloat16; break;
            }
        }
        
        // Create 4D input tensor with proper shape
        torch::Tensor input = torch::randn({batch, channels, height, width}, 
                                           torch::TensorOptions().dtype(dtype));
        
        torch::Tensor output;
        
        switch (call_type) {
            case 0: {
                // Test functional API: torch::channel_shuffle
                output = torch::channel_shuffle(input, groups);
                break;
            }
            
            case 1: {
                // Test with different tensor shapes (3D - treating as NCL format)
                int64_t length = static_cast<int64_t>((offset < Size ? Data[offset++] % 16 : 4) + 1);
                torch::Tensor input_3d = torch::randn({batch, channels, length},
                                                      torch::TensorOptions().dtype(dtype));
                try {
                    output = torch::channel_shuffle(input_3d, groups);
                } catch (...) {
                    // Silently ignore if 3D not supported
                    output = torch::channel_shuffle(input, groups);
                }
                break;
            }
            
            case 2: {
                // Test with different groups values that divide channels
                for (int64_t g = 1; g <= channels && g <= 8; g++) {
                    if (channels % g == 0) {
                        try {
                            torch::Tensor temp_output = torch::channel_shuffle(input, g);
                            // Verify shape preservation
                            if (temp_output.sizes() != input.sizes()) {
                                break;
                            }
                        } catch (...) {
                            // Silently ignore expected failures
                        }
                    }
                }
                // Final call for coverage
                output = torch::channel_shuffle(input, groups);
                break;
            }
        }
        
        // Validate output shape matches input shape
        if (output.defined() && output.sizes() != input.sizes()) {
            std::cerr << "Shape mismatch: input " << input.sizes() 
                      << " vs output " << output.sizes() << std::endl;
        }
        
        // Additional coverage: test with contiguous and non-contiguous tensors
        if (offset < Size && Data[offset] % 2 == 0) {
            // Create non-contiguous tensor via transpose then transpose back
            torch::Tensor non_contig = input.transpose(2, 3).transpose(2, 3);
            try {
                torch::Tensor nc_output = torch::channel_shuffle(non_contig, groups);
                (void)nc_output;
            } catch (...) {
                // Silently ignore - non-contiguous may behave differently
            }
        }
        
        // Additional coverage: test with requires_grad
        if (offset < Size && Data[offset] % 3 == 0) {
            torch::Tensor grad_input = input.clone().set_requires_grad(true);
            try {
                torch::Tensor grad_output = torch::channel_shuffle(grad_input, groups);
                (void)grad_output;
            } catch (...) {
                // Silently ignore
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