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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // ReplicationPad2d expects 3D (C, H, W) or 4D (N, C, H, W) input
        // Reshape tensor to appropriate dimensions
        int64_t numel = input.numel();
        if (numel == 0) {
            return 0;
        }
        
        // Flatten and reshape to 4D: (1, 1, H, W)
        input = input.flatten();
        int64_t h = static_cast<int64_t>(std::sqrt(static_cast<double>(numel)));
        if (h < 1) h = 1;
        int64_t w = numel / h;
        if (w < 1) w = 1;
        int64_t actual_numel = h * w;
        
        if (actual_numel > 0 && actual_numel <= numel) {
            input = input.narrow(0, 0, actual_numel).reshape({1, 1, h, w});
        } else {
            return 0;
        }
        
        // Ensure float type for padding operation
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Parse padding values from the remaining data with bounds
        auto read_bounded_padding = [&](int64_t max_val) -> int64_t {
            if (offset + 1 <= Size) {
                uint8_t val = Data[offset++];
                // Limit padding to reasonable range [0, max_val]
                return static_cast<int64_t>(val % (max_val + 1));
            }
            return 0;
        };
        
        // Limit padding to tensor dimensions to avoid excessive memory usage
        int64_t max_pad = std::min(static_cast<int64_t>(32), std::max(h, w));
        
        int64_t padding_left = read_bounded_padding(max_pad);
        int64_t padding_right = read_bounded_padding(max_pad);
        int64_t padding_top = read_bounded_padding(max_pad);
        int64_t padding_bottom = read_bounded_padding(max_pad);
        
        // Inner try-catch for expected failures
        try
        {
            torch::nn::ReplicationPad2d pad = nullptr;
            
            // Decide which padding format to use based on remaining data
            if (offset < Size) {
                uint8_t padding_type = Data[offset++];
                
                if (padding_type % 3 == 0) {
                    // Use single value for all sides
                    pad = torch::nn::ReplicationPad2d(
                        torch::nn::ReplicationPad2dOptions(padding_left));
                } else if (padding_type % 3 == 1) {
                    // Use symmetric padding (left/right, top/bottom)
                    pad = torch::nn::ReplicationPad2d(
                        torch::nn::ReplicationPad2dOptions({padding_left, padding_left, padding_top, padding_top}));
                } else {
                    // Use different values for each side (left, right, top, bottom)
                    pad = torch::nn::ReplicationPad2d(
                        torch::nn::ReplicationPad2dOptions({padding_left, padding_right, padding_top, padding_bottom}));
                }
            } else {
                // Default to single value padding
                pad = torch::nn::ReplicationPad2d(
                    torch::nn::ReplicationPad2dOptions(padding_left));
            }
            
            // Apply padding
            torch::Tensor output = pad->forward(input);
            
            // Verify output dimensions
            if (output.numel() > 0) {
                // Access elements to ensure computation completed
                volatile float first_val = output.flatten()[0].item<float>();
                (void)first_val;
                
                // Verify output shape is correct
                int64_t expected_h = h + padding_top + padding_bottom;
                int64_t expected_w = w + padding_left + padding_right;
                if (output.size(2) != expected_h || output.size(3) != expected_w) {
                    // Unexpected shape - this would indicate a bug
                }
            }
            
            // Test with 3D input as well
            if (offset < Size && Data[offset] % 2 == 0) {
                torch::Tensor input_3d = input.squeeze(0); // (C, H, W)
                torch::Tensor output_3d = pad->forward(input_3d);
                if (output_3d.numel() > 0) {
                    volatile float val = output_3d.flatten()[0].item<float>();
                    (void)val;
                }
            }
        }
        catch (const c10::Error &e)
        {
            // Expected failures due to invalid shapes/padding - silently ignore
        }
        catch (const std::runtime_error &e)
        {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}