#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    // Minimum size needed: 1 byte for downscale_factor + tensor metadata
    if (size < 3) {
        return 0;  // Not enough data to work with
    }

    try
    {
        size_t offset = 0;

        // Parse downscale_factor from first byte
        uint8_t downscale_byte = data[offset++];
        // Limit downscale_factor to reasonable range [1, 8] to avoid extreme memory usage
        int downscale_factor = (downscale_byte % 8) + 1;

#ifdef DEBUG_FUZZ
        std::cout << "Downscale factor: " << downscale_factor << std::endl;
#endif

        // Create input tensor from remaining bytes
        torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);

        // pixel_unshuffle requires at least 3 dimensions (batch/channel, height, width)
        // If tensor has fewer dimensions, reshape or pad dimensions
        if (input.dim() < 3) {
            // Reshape to add dimensions
            std::vector<int64_t> new_shape;
            
            if (input.dim() == 0) {
                // Scalar: reshape to [1, 1, 1]
                new_shape = {1, 1, 1};
            } else if (input.dim() == 1) {
                // 1D: reshape to [1, 1, N]
                new_shape = {1, 1, input.size(0)};
            } else if (input.dim() == 2) {
                // 2D: reshape to [1, H, W]
                new_shape = {1, input.size(0), input.size(1)};
            }
            
            input = input.reshape(new_shape);
        }

        // For 5D or higher tensors, we need to handle them specially
        // pixel_unshuffle expects (*, C, H*r, W*r) format
        if (input.dim() > 4) {
            // Flatten batch dimensions
            auto sizes = input.sizes();
            int64_t batch_size = 1;
            for (int i = 0; i < input.dim() - 3; ++i) {
                batch_size *= sizes[i];
            }
            input = input.reshape({batch_size, sizes[input.dim()-3], sizes[input.dim()-2], sizes[input.dim()-1]});
        }

#ifdef DEBUG_FUZZ
        std::cout << "Input tensor shape: " << input.sizes() << std::endl;
        std::cout << "Input tensor dtype: " << input.dtype() << std::endl;
        std::cout << "Input tensor device: " << input.device() << std::endl;
#endif

        // Try different tensor memory layouts
        if (offset < size && data[offset] % 3 == 0) {
            // Make tensor non-contiguous by transposing and back
            if (input.dim() >= 2) {
                auto last_two = input.dim() - 1;
                auto second_last = input.dim() - 2;
                input = input.transpose(second_last, last_two).transpose(second_last, last_two);
            }
        }

        // Apply pixel_unshuffle operation
        torch::Tensor output;
        
        try {
            // Main operation under test
            output = torch::nn::functional::pixel_unshuffle(input, 
                torch::nn::functional::PixelUnshuffleFuncOptions(downscale_factor));

#ifdef DEBUG_FUZZ
            std::cout << "Output tensor shape: " << output.sizes() << std::endl;
            std::cout << "Output tensor dtype: " << output.dtype() << std::endl;
#endif

            // Verify output properties
            // Expected: (*, C * r^2, H, W) where input was (*, C, H * r, W * r)
            auto in_sizes = input.sizes();
            auto out_sizes = output.sizes();
            
            // Basic shape validation (without early return to allow edge cases)
            if (input.dim() >= 3) {
                int64_t r2 = downscale_factor * downscale_factor;
                int64_t expected_channels = in_sizes[in_sizes.size() - 3] * r2;
                int64_t actual_channels = out_sizes[out_sizes.size() - 3];
                
                if (expected_channels != actual_channels && in_sizes[in_sizes.size()-2] % downscale_factor == 0) {
#ifdef DEBUG_FUZZ
                    std::cout << "Warning: Channel mismatch. Expected: " << expected_channels 
                             << ", Got: " << actual_channels << std::endl;
#endif
                }
            }

            // Test with different downscale factors if we have more data
            if (offset + 1 < size) {
                uint8_t second_factor_byte = data[offset++];
                int second_factor = (second_factor_byte % 4) + 1;
                
                // Try chaining operations (unshuffle then shuffle back)
                try {
                    auto unshuffled = torch::nn::functional::pixel_unshuffle(input,
                        torch::nn::functional::PixelUnshuffleFuncOptions(second_factor));
                    
                    // Only try to shuffle back if dimensions are compatible
                    if (unshuffled.size(-1) > 0 && unshuffled.size(-2) > 0) {
                        auto reshuffled = torch::nn::functional::pixel_shuffle(unshuffled, second_factor);
#ifdef DEBUG_FUZZ
                        std::cout << "Round-trip test completed" << std::endl;
#endif
                    }
                } catch (const c10::Error& e) {
                    // Expected for incompatible dimensions
#ifdef DEBUG_FUZZ
                    std::cout << "Round-trip failed (expected for some inputs): " << e.what() << std::endl;
#endif
                }
            }

            // Test edge cases with zero-sized dimensions
            if (offset + 1 < size && data[offset] % 10 == 0) {
                try {
                    auto zero_tensor = torch::zeros({1, 1, 0, 4});
                    auto zero_result = torch::nn::functional::pixel_unshuffle(zero_tensor,
                        torch::nn::functional::PixelUnshuffleFuncOptions(2));
#ifdef DEBUG_FUZZ
                    std::cout << "Zero-dimension test passed" << std::endl;
#endif
                } catch (const c10::Error& e) {
                    // Some configurations might not support zero dimensions
#ifdef DEBUG_FUZZ
                    std::cout << "Zero-dimension test failed: " << e.what() << std::endl;
#endif
                }
            }

            // Test with different memory formats if supported
            if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                try {
                    // Try channels_last memory format for 4D tensors
                    if (input.dim() == 4 && input.size(0) > 0 && input.size(1) > 0 && 
                        input.size(2) > 0 && input.size(3) > 0) {
                        auto channels_last_input = input.to(torch::MemoryFormat::ChannelsLast);
                        auto cl_output = torch::nn::functional::pixel_unshuffle(channels_last_input,
                            torch::nn::functional::PixelUnshuffleFuncOptions(downscale_factor));
#ifdef DEBUG_FUZZ
                        std::cout << "Channels-last format test completed" << std::endl;
#endif
                    }
                } catch (const c10::Error& e) {
#ifdef DEBUG_FUZZ
                    std::cout << "Channels-last test failed: " << e.what() << std::endl;
#endif
                }
            }

        } catch (const c10::Error& e) {
            // PyTorch-specific errors (dimension mismatches, etc.)
#ifdef DEBUG_FUZZ
            std::cout << "PyTorch error during pixel_unshuffle: " << e.what() << std::endl;
#endif
            // Don't discard - these are interesting for finding edge cases
            return 0;
        }

    }
    catch (const std::runtime_error& e)
    {
        // Errors from tensor creation
#ifdef DEBUG_FUZZ
        std::cout << "Runtime error: " << e.what() << std::endl;
#endif
        return 0;  // Keep going, these might reveal parsing issues
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // Discard only for unexpected exceptions
    }
    
    return 0;
}