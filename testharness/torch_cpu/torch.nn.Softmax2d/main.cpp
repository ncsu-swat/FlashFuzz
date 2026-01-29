#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty tensors
        if (input.numel() == 0) {
            return 0;
        }
        
        // Softmax2d requires a 4D tensor with shape [N, C, H, W]
        // or 3D tensor with shape [C, H, W]
        int64_t total_elements = input.numel();
        
        // Use fuzzer data to determine tensor shape if available
        int64_t batch_size = 1;
        int64_t channels = 1;
        int64_t height = 2;
        int64_t width = 2;
        
        if (offset < Size) {
            // Use fuzzer byte to determine if we should use 3D or 4D
            bool use_3d = (Data[offset % Size] % 2) == 0;
            offset++;
            
            if (use_3d) {
                // 3D input [C, H, W]
                // Calculate dimensions that multiply to total_elements
                channels = std::max(int64_t(1), std::min(int64_t(16), total_elements));
                int64_t remaining = total_elements / channels;
                
                if (remaining > 0) {
                    height = std::max(int64_t(1), (int64_t)std::sqrt(remaining));
                    width = remaining / height;
                    
                    // Ensure we use all elements
                    int64_t actual = channels * height * width;
                    if (actual != total_elements) {
                        // Fall back to simple reshape
                        channels = total_elements;
                        height = 1;
                        width = 1;
                    }
                } else {
                    height = 1;
                    width = 1;
                }
                
                try {
                    input = input.reshape({channels, height, width});
                } catch (...) {
                    return 0;
                }
            } else {
                // 4D input [N, C, H, W]
                // Calculate dimensions that multiply to total_elements
                batch_size = std::max(int64_t(1), std::min(int64_t(4), total_elements));
                int64_t remaining = total_elements / batch_size;
                
                if (remaining > 0) {
                    channels = std::max(int64_t(1), std::min(int64_t(16), remaining));
                    remaining = remaining / channels;
                    
                    if (remaining > 0) {
                        height = std::max(int64_t(1), (int64_t)std::sqrt(remaining));
                        width = remaining / height;
                    } else {
                        height = 1;
                        width = 1;
                    }
                } else {
                    channels = 1;
                    height = 1;
                    width = 1;
                }
                
                // Ensure we use all elements
                int64_t actual = batch_size * channels * height * width;
                if (actual != total_elements) {
                    // Fall back to simple 4D reshape
                    batch_size = total_elements;
                    channels = 1;
                    height = 1;
                    width = 1;
                }
                
                try {
                    input = input.reshape({batch_size, channels, height, width});
                } catch (...) {
                    return 0;
                }
            }
        } else {
            // Default: reshape to 4D
            try {
                input = input.reshape({1, total_elements, 1, 1});
            } catch (...) {
                return 0;
            }
        }
        
        // Ensure input is floating point (Softmax requires float)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create Softmax2d module
        torch::nn::Softmax2d softmax2d;
        
        // Apply Softmax2d to the input tensor
        torch::Tensor output = softmax2d(input);
        
        // Verify the output
        if (output.defined()) {
            float sum = output.sum().item<float>();
            
            // Prevent compiler optimization
            if (std::isnan(sum)) {
                return 0;
            }
            
            // Additional verification: check output shape matches input
            if (output.sizes() != input.sizes()) {
                std::cerr << "Shape mismatch in output" << std::endl;
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