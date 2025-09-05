#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for upscale_factor
        if (Size < 1) {
            return 0;
        }
        
        // Parse upscale_factor from fuzzer input
        // pixel_shuffle requires positive integer upscale_factor
        // Let's limit it to reasonable range [1, 16] to avoid memory issues
        uint8_t factor_byte = Data[offset++];
        int64_t upscale_factor = 1 + (factor_byte % 16);
        
        // Create input tensor from fuzzer data
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create a valid tensor, skip this input
            return 0;
        }
        
        // pixel_shuffle has specific requirements:
        // - Input must be at least 3D (batch, channels, height, width) or 4D
        // - channels dimension must be divisible by upscale_factor²
        
        // Try different tensor configurations based on remaining fuzzer bytes
        if (offset < Size) {
            uint8_t config_byte = Data[offset++];
            
            // Strategy 1: Reshape tensor to valid dimensions if possible
            if (config_byte & 0x01) {
                try {
                    // Ensure we have at least 3 dimensions
                    if (input.dim() < 3) {
                        // Add dimensions
                        while (input.dim() < 3) {
                            input = input.unsqueeze(0);
                        }
                    }
                    
                    // Ensure channels dimension is divisible by upscale_factor²
                    if (input.dim() >= 3) {
                        int64_t channels_dim_idx = input.dim() - 3;
                        if (channels_dim_idx >= 0) {
                            int64_t current_channels = input.size(channels_dim_idx);
                            int64_t factor_squared = upscale_factor * upscale_factor;
                            
                            if (current_channels % factor_squared != 0) {
                                // Pad or reshape to make it divisible
                                int64_t new_channels = ((current_channels / factor_squared) + 1) * factor_squared;
                                auto sizes = input.sizes().vec();
                                sizes[channels_dim_idx] = new_channels;
                                
                                // Try to reshape if total elements allow
                                int64_t total_elements = input.numel();
                                int64_t new_total = 1;
                                for (auto s : sizes) {
                                    new_total *= s;
                                }
                                
                                if (total_elements >= new_total) {
                                    input = input.flatten().narrow(0, 0, new_total).reshape(sizes);
                                } else {
                                    // Pad with zeros
                                    std::vector<int64_t> pad_vec;
                                    for (int i = input.dim() - 1; i >= 0; i--) {
                                        if (i == channels_dim_idx) {
                                            pad_vec.push_back(0);
                                            pad_vec.push_back(new_channels - current_channels);
                                        } else {
                                            pad_vec.push_back(0);
                                            pad_vec.push_back(0);
                                        }
                                    }
                                    input = torch::nn::functional::pad(input, 
                                        torch::nn::functional::PadFuncOptions(pad_vec));
                                }
                            }
                        }
                    }
                } catch (...) {
                    // Continue with original tensor
                }
            }
            
            // Strategy 2: Create specific test shapes
            if (config_byte & 0x02) {
                try {
                    int64_t factor_sq = upscale_factor * upscale_factor;
                    // Create a tensor with valid shape for pixel_shuffle
                    int64_t batch = 1 + (config_byte & 0x03);
                    int64_t channels = factor_sq * (1 + ((config_byte >> 2) & 0x03));
                    int64_t height = 2 + ((config_byte >> 4) & 0x07);
                    int64_t width = 2 + ((config_byte >> 6) & 0x03);
                    
                    auto options = torch::TensorOptions().dtype(input.dtype());
                    input = torch::randn({batch, channels, height, width}, options);
                } catch (...) {
                    // Keep original tensor
                }
            }
        }
        
        // Apply pixel_shuffle with various configurations
        torch::Tensor output;
        
        try {
            // Main operation
            output = torch::pixel_shuffle(input, upscale_factor);
            
            // Verify output properties
            if (output.defined()) {
                // Check basic properties
                auto out_shape = output.sizes();
                auto in_shape = input.sizes();
                
                // Additional operations on the output to increase coverage
                if (offset < Size && Data[offset++] & 0x01) {
                    // Try inverse operation (pixel_unshuffle if available)
                    try {
                        auto reconstructed = torch::pixel_unshuffle(output, upscale_factor);
                        // Compare shapes
                        if (reconstructed.sizes() != input.sizes()) {
                            // This might indicate an issue but don't crash
                        }
                    } catch (...) {
                        // pixel_unshuffle might not be available or might fail
                    }
                }
                
                // Test with different memory layouts
                if (offset < Size && Data[offset++] & 0x01) {
                    try {
                        // Test with non-contiguous tensor
                        auto transposed = input.transpose(-1, -2);
                        auto out_transposed = torch::pixel_shuffle(transposed.contiguous(), upscale_factor);
                    } catch (...) {
                        // Ignore failures on edge cases
                    }
                }
                
                // Test with different devices if available
                if (torch::cuda::is_available() && offset < Size && Data[offset++] & 0x01) {
                    try {
                        auto cuda_input = input.to(torch::kCUDA);
                        auto cuda_output = torch::pixel_shuffle(cuda_input, upscale_factor);
                        // Move back to CPU for comparison
                        auto cpu_output = cuda_output.to(torch::kCPU);
                    } catch (...) {
                        // CUDA operations might fail
                    }
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue fuzzing
            return 0;
        } catch (const std::exception& e) {
            // Log unexpected errors but continue
            std::cout << "Exception caught: " << e.what() << std::endl;
            return 0;
        }
        
        // Test edge cases with extreme upscale factors
        if (offset < Size) {
            uint8_t extreme_test = Data[offset++];
            if (extreme_test & 0x01) {
                try {
                    // Test with upscale_factor = 1 (should be identity-like)
                    auto identity_out = torch::pixel_shuffle(input, 1);
                } catch (...) {}
            }
            
            if (extreme_test & 0x02) {
                try {
                    // Test with larger upscale_factor if tensor allows
                    int64_t large_factor = 2 + (extreme_test >> 2) % 8;
                    if (input.dim() >= 3) {
                        int64_t channels = input.size(input.dim() - 3);
                        if (channels >= large_factor * large_factor) {
                            auto large_out = torch::pixel_shuffle(input, large_factor);
                        }
                    }
                } catch (...) {}
            }
        }
        
        // Test with different data types
        if (offset < Size && Data[offset++] & 0x01) {
            try {
                // Convert to different dtype and test
                auto float_input = input.to(torch::kFloat32);
                auto float_output = torch::pixel_shuffle(float_input, upscale_factor);
                
                if (offset < Size && Data[offset++] & 0x01) {
                    auto half_input = input.to(torch::kHalf);
                    auto half_output = torch::pixel_shuffle(half_input, upscale_factor);
                }
            } catch (...) {
                // Type conversion might fail for some types
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}