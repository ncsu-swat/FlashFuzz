#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to consume a value from fuzzer data
template<typename T>
T consumeValue(const uint8_t* data, size_t& offset, size_t size, T min_val, T max_val) {
    if (offset + sizeof(T) > size) {
        offset = size;
        return min_val;
    }
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    
    // Ensure value is in range [min_val, max_val]
    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;
    return value;
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check - need at least some bytes for tensor creation and groups parameter
        if (Size < 10) {
            return 0;
        }

        // Create input tensor - native_channel_shuffle typically expects 4D tensors
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try to create a default one
            if (offset < Size) {
                uint8_t shape_selector = Data[offset++];
                int64_t n = 1 + (shape_selector & 0x07);  // batch size 1-8
                int64_t c = 1 + ((shape_selector >> 3) & 0x1F);  // channels 1-32
                int64_t h = 1 + ((Data[offset % Size] & 0x0F));  // height 1-16
                int64_t w = 1 + ((Data[(offset + 1) % Size] & 0x0F));  // width 1-16
                offset += 2;
                
                auto dtype = fuzzer_utils::parseDataType(Data[offset % Size]);
                offset++;
                
                input = torch::randn({n, c, h, w}, torch::TensorOptions().dtype(dtype));
            } else {
                return 0;
            }
        }

        // Consume groups parameter
        int64_t groups = 1;
        if (offset < Size) {
            uint8_t groups_byte = Data[offset++];
            // Map to reasonable range: 1 to 32
            groups = 1 + (groups_byte % 32);
        }

        // Try different tensor manipulations to increase coverage
        if (offset < Size && Data[offset++] % 4 == 0) {
            // Sometimes make the tensor non-contiguous
            if (input.dim() >= 2) {
                input = input.transpose(0, 1).transpose(0, 1);
            }
        }

        // Reshape input to 4D if it's not already
        if (input.dim() != 4) {
            // Calculate total elements
            int64_t total = input.numel();
            if (total == 0) {
                // Create a small 4D tensor for empty case
                input = torch::zeros({1, groups, 1, 1}, input.options());
            } else {
                // Try to reshape to 4D
                // Find factors for reshaping
                int64_t n = 1, c = groups, h = 1, w = 1;
                
                // Try to make channels divisible by groups
                if (total >= groups) {
                    c = groups;
                    int64_t remaining = total / c;
                    
                    // Distribute remaining dimensions
                    if (remaining > 0) {
                        n = 1 + (remaining % 4);
                        remaining /= n;
                        if (remaining > 0) {
                            h = 1 + (remaining % 8);
                            w = remaining / h;
                            if (w == 0) w = 1;
                        }
                    }
                    
                    // Adjust if product doesn't match
                    while (n * c * h * w < total && w < total) {
                        w++;
                    }
                    while (n * c * h * w > total && w > 1) {
                        w--;
                    }
                    
                    // Final adjustment
                    if (n * c * h * w != total) {
                        // Fall back to simple reshape
                        c = 1;
                        while (c <= groups && total % c != 0) {
                            c++;
                        }
                        if (c > groups || total % c != 0) {
                            c = groups;
                        }
                        n = 1;
                        h = 1;
                        w = total / (n * c * h);
                        if (w == 0 || n * c * h * w != total) {
                            // Can't reshape properly, create new tensor
                            input = torch::randn({1, groups, 2, 2}, input.options());
                        } else {
                            input = input.reshape({n, c, h, w});
                        }
                    } else {
                        input = input.reshape({n, c, h, w});
                    }
                } else {
                    // Total elements less than groups, create a proper tensor
                    input = torch::randn({1, groups, 2, 2}, input.options());
                }
            }
        }

        // Ensure channels are divisible by groups (required for native_channel_shuffle)
        if (input.size(1) % groups != 0) {
            // Adjust channels to be divisible by groups
            int64_t new_c = groups * (std::max(int64_t(1), input.size(1) / groups));
            if (new_c == 0) new_c = groups;
            
            // Resize or pad the tensor
            if (input.numel() > 0) {
                auto sizes = input.sizes().vec();
                sizes[1] = new_c;
                input = torch::randn(sizes, input.options());
            } else {
                input = torch::randn({1, new_c, 2, 2}, input.options());
            }
        }

        // Additional tensor manipulations based on fuzzer input
        if (offset < Size) {
            uint8_t manipulation = Data[offset++];
            
            switch (manipulation % 8) {
                case 0:
                    // Make tensor require gradient
                    if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble ||
                        input.dtype() == torch::kHalf || input.dtype() == torch::kBFloat16) {
                        input = input.requires_grad_(true);
                    }
                    break;
                case 1:
                    // Pin memory if CPU tensor
                    if (!input.is_cuda() && input.dtype() != torch::kBool) {
                        try {
                            input = input.pin_memory();
                        } catch (...) {
                            // Ignore pin_memory failures
                        }
                    }
                    break;
                case 2:
                    // Make sparse if possible (though native_channel_shuffle likely doesn't support sparse)
                    // Skip sparse for this op as it's not typically supported
                    break;
                case 3:
                    // Change to different memory format
                    try {
                        input = input.contiguous(torch::MemoryFormat::ChannelsLast);
                    } catch (...) {
                        // Ignore memory format failures
                    }
                    break;
                case 4:
                    // Create a view with different strides
                    if (input.numel() > 0 && input.dim() == 4) {
                        try {
                            input = input.permute({0, 1, 3, 2}).permute({0, 1, 3, 2});
                        } catch (...) {
                            // Ignore permute failures
                        }
                    }
                    break;
                default:
                    // Keep tensor as is
                    break;
            }
        }

        // Call native_channel_shuffle
        torch::Tensor output;
        try {
            // torch::native_channel_shuffle is typically called through the functional API
            // The actual C++ API might be torch::channel_shuffle or in the native namespace
            output = torch::native::channel_shuffle(input, groups);
            
            // Verify output properties
            if (output.defined()) {
                // Check output shape matches input shape
                if (output.sizes() != input.sizes()) {
                    std::cerr << "Warning: Output shape mismatch" << std::endl;
                }
                
                // Try backward pass if gradient is enabled
                if (output.requires_grad()) {
                    try {
                        auto grad_output = torch::ones_like(output);
                        output.backward(grad_output);
                    } catch (const std::exception& e) {
                        // Backward pass failed, but that's okay for fuzzing
                    }
                }
                
                // Additional operations to increase coverage
                if (offset < Size && Data[offset++] % 2 == 0) {
                    // Try inverse shuffle (shuffle again with same groups should work)
                    try {
                        auto output2 = torch::native::channel_shuffle(output, groups);
                        // Could compare output2 with some permutation of input
                    } catch (...) {
                        // Ignore failures in secondary operations
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch C10 errors are expected for invalid inputs
            // Continue fuzzing
        } catch (const std::exception& e) {
            // Log unexpected exceptions but continue
            std::cerr << "Unexpected exception in channel_shuffle: " << e.what() << std::endl;
        }

        // Try edge cases with different group values
        if (offset < Size && Data[offset++] % 3 == 0) {
            std::vector<int64_t> test_groups = {1, input.size(1), input.size(1)/2, 0, -1, INT64_MAX};
            for (auto g : test_groups) {
                try {
                    auto test_output = torch::native::channel_shuffle(input, g);
                } catch (...) {
                    // Expected to fail for invalid groups
                }
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}