#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes: 2 for tensor metadata + 1 for groups
        if (Size < 3) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create a valid tensor, skip this input
            return 0;
        }

        // Ensure we have at least one more byte for groups
        if (offset >= Size) {
            return 0;
        }

        // Parse groups parameter
        uint8_t groups_byte = Data[offset++];
        // Groups must be positive and should divide the number of channels
        // Let's limit groups to a reasonable range [1, 256]
        int64_t groups = static_cast<int64_t>((groups_byte % 256) + 1);

        // native_channel_shuffle expects at least 2D tensor (batch, channels, ...)
        // But it can work with various dimensions as long as there's a channel dimension
        
        // Try different tensor manipulations to increase coverage
        if (offset < Size && Data[offset] % 4 == 0) {
            // Sometimes reshape the tensor to ensure it has proper dimensions
            offset++;
            
            // Calculate a valid shape based on tensor's numel
            int64_t numel = input.numel();
            if (numel > 0) {
                // Try to create a 4D tensor (batch, channels, height, width)
                int64_t batch = 1;
                int64_t channels = groups * ((numel / groups) > 0 ? 1 : 1);
                int64_t remaining = numel / (batch * channels);
                
                if (remaining > 0 && numel % (batch * channels) == 0) {
                    // Split remaining into height and width
                    int64_t height = 1;
                    int64_t width = remaining;
                    
                    // Try to make height and width more balanced
                    for (int64_t h = 2; h * h <= remaining; h++) {
                        if (remaining % h == 0) {
                            height = h;
                            width = remaining / h;
                        }
                    }
                    
                    try {
                        input = input.reshape({batch, channels, height, width});
                    } catch (...) {
                        // If reshape fails, continue with original tensor
                    }
                }
            }
        }

        // Ensure tensor has at least 2 dimensions for channel_shuffle
        if (input.dim() < 2) {
            // Add dimensions to make it at least 2D
            while (input.dim() < 2) {
                input = input.unsqueeze(0);
            }
        }

        // Adjust groups to be valid for the number of channels
        if (input.dim() >= 2) {
            int64_t num_channels = input.size(1);
            if (num_channels > 0) {
                // Groups must divide the number of channels evenly
                // Find a valid divisor
                if (num_channels % groups != 0) {
                    // Find the closest valid divisor
                    for (int64_t g = groups; g > 0; g--) {
                        if (num_channels % g == 0) {
                            groups = g;
                            break;
                        }
                    }
                }
            }
        }

        // Try different tensor properties based on fuzzer input
        if (offset < Size) {
            uint8_t property_selector = Data[offset++];
            
            switch (property_selector % 8) {
                case 0:
                    // Make tensor contiguous
                    input = input.contiguous();
                    break;
                case 1:
                    // Transpose if possible
                    if (input.dim() >= 2) {
                        try {
                            input = input.transpose(0, 1);
                        } catch (...) {}
                    }
                    break;
                case 2:
                    // Add batch dimension if not present
                    if (input.dim() == 2) {
                        input = input.unsqueeze(0);
                    }
                    break;
                case 3:
                    // Try to make it non-contiguous
                    if (input.dim() >= 2 && input.size(0) > 1) {
                        try {
                            input = input.narrow(0, 0, input.size(0));
                        } catch (...) {}
                    }
                    break;
                case 4:
                    // Convert to different dtype if requested
                    if (offset < Size) {
                        uint8_t dtype_change = Data[offset++];
                        if (dtype_change % 4 == 0) {
                            try {
                                input = input.to(torch::kFloat32);
                            } catch (...) {}
                        }
                    }
                    break;
                case 5:
                    // Try requires_grad
                    if (input.dtype() == torch::kFloat32 || input.dtype() == torch::kFloat64) {
                        try {
                            input = input.requires_grad_(true);
                        } catch (...) {}
                    }
                    break;
                case 6:
                    // Try to use CUDA if available
                    if (torch::cuda::is_available() && offset < Size && Data[offset++] % 10 == 0) {
                        try {
                            input = input.cuda();
                        } catch (...) {
                            // CUDA operation failed, continue with CPU tensor
                        }
                    }
                    break;
                case 7:
                    // Create a view with different strides
                    if (input.dim() >= 2 && input.numel() > 0) {
                        try {
                            input = input.as_strided(input.sizes(), input.strides());
                        } catch (...) {}
                    }
                    break;
            }
        }

        // Call native_channel_shuffle
        torch::Tensor output;
        try {
            // The actual function call
            output = torch::native_channel_shuffle(input, groups);
            
            // Verify output properties for additional coverage
            if (output.defined()) {
                // Check basic properties
                bool same_shape = (output.sizes() == input.sizes());
                bool same_dtype = (output.dtype() == input.dtype());
                bool same_device = (output.device() == input.device());
                
                // Access some elements to trigger potential memory issues
                if (output.numel() > 0) {
                    try {
                        auto first_elem = output.flatten()[0];
                        if (output.numel() > 1) {
                            auto last_elem = output.flatten()[output.numel() - 1];
                        }
                    } catch (...) {
                        // Element access failed, but continue
                    }
                }
                
                // Try backward pass if applicable
                if (output.requires_grad() && output.dtype().isFloatingPoint()) {
                    try {
                        auto loss = output.sum();
                        loss.backward();
                    } catch (...) {
                        // Backward pass failed, but continue
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue fuzzing
            return 0;
        } catch (const std::exception& e) {
            // Log unexpected exceptions for debugging
            std::cout << "Exception caught: " << e.what() << std::endl;
            return -1;
        }

        // Additional operations to increase coverage
        if (output.defined() && offset < Size) {
            uint8_t extra_ops = Data[offset++];
            
            switch (extra_ops % 4) {
                case 0:
                    // Try channel_shuffle again with different groups
                    if (offset < Size) {
                        int64_t new_groups = (Data[offset++] % 16) + 1;
                        if (output.dim() >= 2) {
                            int64_t channels = output.size(1);
                            // Find valid divisor
                            for (int64_t g = new_groups; g > 0; g--) {
                                if (channels % g == 0) {
                                    new_groups = g;
                                    break;
                                }
                            }
                            try {
                                auto output2 = torch::native_channel_shuffle(output, new_groups);
                            } catch (...) {}
                        }
                    }
                    break;
                case 1:
                    // Compare with manual channel shuffle implementation
                    if (input.dim() >= 2 && groups > 1) {
                        try {
                            int64_t batch = input.size(0);
                            int64_t channels = input.size(1);
                            if (channels % groups == 0) {
                                int64_t channels_per_group = channels / groups;
                                auto reshaped = input.view({batch, groups, channels_per_group});
                                for (int64_t i = 2; i < input.dim(); i++) {
                                    reshaped = reshaped.unsqueeze(-1);
                                }
                                auto transposed = reshaped.transpose(1, 2);
                                auto manual_output = transposed.contiguous().view(input.sizes());
                            }
                        } catch (...) {}
                    }
                    break;
                case 2:
                    // Try inverse operation
                    if (output.dim() >= 2) {
                        try {
                            // Channel shuffle is its own inverse when applied twice with same groups
                            auto inverse = torch::native_channel_shuffle(output, groups);
                            // Could compare with original input here
                        } catch (...) {}
                    }
                    break;
                case 3:
                    // Test with edge case groups
                    if (output.dim() >= 2) {
                        int64_t channels = output.size(1);
                        try {
                            // Groups = 1 (no shuffle)
                            auto no_shuffle = torch::native_channel_shuffle(output, 1);
                            // Groups = channels (maximum shuffle)
                            if (channels > 0) {
                                auto max_shuffle = torch::native_channel_shuffle(output, channels);
                            }
                        } catch (...) {}
                    }
                    break;
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