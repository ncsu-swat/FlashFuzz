#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 3) {
            // Need at least: tensor metadata (2 bytes) + upscale_factor (1 byte)
            return 0;
        }

        size_t offset = 0;

        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse upscale_factor from remaining data
        int upscale_factor = 1;
        if (offset < Size) {
            uint8_t factor_byte = Data[offset++];
            // Limit upscale_factor to reasonable range [1, 16] to avoid memory issues
            upscale_factor = (factor_byte % 16) + 1;
        }

#ifdef DEBUG_FUZZ
        std::cout << "Input tensor shape: " << input.sizes() << std::endl;
        std::cout << "Input tensor dtype: " << input.dtype() << std::endl;
        std::cout << "Upscale factor: " << upscale_factor << std::endl;
#endif

        // pixel_shuffle expects input of shape (*, C * r^2, H, W)
        // We need at least 3 dimensions for meaningful operation
        if (input.dim() < 3) {
            // Reshape to add dimensions if needed
            if (input.dim() == 0) {
                input = input.reshape({1, 1, 1, 1});
            } else if (input.dim() == 1) {
                int64_t size = input.size(0);
                input = input.reshape({1, size, 1, 1});
            } else if (input.dim() == 2) {
                input = input.reshape({1, input.size(0), input.size(1), 1});
            }
        }

        // Ensure we have 4 dimensions for pixel_shuffle
        if (input.dim() == 3) {
            input = input.unsqueeze(0);
        }

        // Adjust channel dimension to be divisible by upscale_factor^2
        if (input.dim() >= 2) {
            int64_t channels = input.size(-3);
            int64_t required_factor = upscale_factor * upscale_factor;
            
            if (channels % required_factor != 0) {
                // Pad or adjust channels to make it divisible
                int64_t new_channels = ((channels / required_factor) + 1) * required_factor;
                if (new_channels > channels && new_channels <= 1024) {
                    // Pad with zeros to reach required channel count
                    std::vector<int64_t> new_shape = input.sizes().vec();
                    new_shape[input.dim() - 3] = new_channels;
                    torch::Tensor padded = torch::zeros(new_shape, input.options());
                    
                    // Copy original data
                    if (input.dim() == 4) {
                        padded.index({torch::indexing::Slice(), 
                                     torch::indexing::Slice(0, channels),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice()}) = input;
                    }
                    input = padded;
                } else if (channels > required_factor) {
                    // Truncate channels to be divisible
                    int64_t new_channels = (channels / required_factor) * required_factor;
                    if (input.dim() == 4) {
                        input = input.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(0, new_channels),
                                           torch::indexing::Slice(),
                                           torch::indexing::Slice()});
                    }
                }
            }
        }

        // Try different tensor configurations
        std::vector<torch::Tensor> test_tensors;
        test_tensors.push_back(input);
        
        // Try with different memory layouts
        if (input.is_contiguous()) {
            test_tensors.push_back(input.permute({0, 1, 3, 2}).contiguous().permute({0, 1, 3, 2}));
        }
        
        // Try with requires_grad
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble || 
            input.dtype() == torch::kHalf || input.dtype() == torch::kBFloat16) {
            torch::Tensor grad_input = input.clone().requires_grad_(true);
            test_tensors.push_back(grad_input);
        }

        for (auto& test_input : test_tensors) {
            try {
                // Main operation: pixel_shuffle
                torch::Tensor output = torch::nn::functional::pixel_shuffle(test_input, 
                    torch::nn::functional::PixelShuffleFuncOptions(upscale_factor));
                
#ifdef DEBUG_FUZZ
                std::cout << "Output shape: " << output.sizes() << std::endl;
                std::cout << "Output dtype: " << output.dtype() << std::endl;
#endif

                // Verify output dimensions
                if (test_input.dim() == 4) {
                    int64_t expected_h = test_input.size(2) * upscale_factor;
                    int64_t expected_w = test_input.size(3) * upscale_factor;
                    int64_t expected_c = test_input.size(1) / (upscale_factor * upscale_factor);
                    
                    if (output.size(2) != expected_h || output.size(3) != expected_w || 
                        output.size(1) != expected_c) {
#ifdef DEBUG_FUZZ
                        std::cerr << "Unexpected output dimensions!" << std::endl;
#endif
                    }
                }

                // Test inverse operation (pixel_unshuffle)
                try {
                    torch::Tensor reconstructed = torch::nn::functional::pixel_unshuffle(output,
                        torch::nn::functional::PixelUnshuffleFuncOptions(upscale_factor));
                    
                    // Check if we can recover the original
                    if (reconstructed.sizes() != test_input.sizes()) {
#ifdef DEBUG_FUZZ
                        std::cerr << "Inverse operation shape mismatch!" << std::endl;
#endif
                    }
                } catch (const c10::Error& e) {
                    // Inverse operation might fail for certain configurations
#ifdef DEBUG_FUZZ
                    std::cout << "Pixel unshuffle failed: " << e.what() << std::endl;
#endif
                }

                // Test with different upscale factors if we have enough channels
                if (test_input.size(1) >= 16 && offset < Size) {
                    for (int alt_factor = 2; alt_factor <= 4; alt_factor++) {
                        if (test_input.size(1) % (alt_factor * alt_factor) == 0) {
                            try {
                                torch::Tensor alt_output = torch::nn::functional::pixel_shuffle(test_input,
                                    torch::nn::functional::PixelShuffleFuncOptions(alt_factor));
#ifdef DEBUG_FUZZ
                                std::cout << "Alternative factor " << alt_factor << " output: " 
                                         << alt_output.sizes() << std::endl;
#endif
                            } catch (const c10::Error& e) {
                                // Some factors might not work
                            }
                        }
                    }
                }

            } catch (const c10::Error& e) {
#ifdef DEBUG_FUZZ
                std::cout << "PyTorch error during pixel_shuffle: " << e.what() << std::endl;
#endif
                // Continue with next test tensor
            }
        }

        // Test edge cases with manually created tensors
        try {
            // Test with minimal valid input
            torch::Tensor minimal = torch::ones({1, 4, 2, 2}, input.options());
            torch::Tensor minimal_out = torch::nn::functional::pixel_shuffle(minimal,
                torch::nn::functional::PixelShuffleFuncOptions(2));
            
            // Test with larger upscale factor
            torch::Tensor large = torch::ones({1, 25, 3, 3}, input.options());
            torch::Tensor large_out = torch::nn::functional::pixel_shuffle(large,
                torch::nn::functional::PixelShuffleFuncOptions(5));
            
        } catch (const c10::Error& e) {
#ifdef DEBUG_FUZZ
            std::cout << "Edge case testing failed: " << e.what() << std::endl;
#endif
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const c10::Error &e)
    {
        std::cout << "PyTorch exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}