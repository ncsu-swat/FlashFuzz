#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Early exit for very small inputs
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 5 dimensions (N, C, D, H, W) for Conv3d
        if (input.dim() < 5) {
            // Reshape to 5D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar to 5D
                new_shape = {1, 1, 1, 1, 1};
            } else if (input.dim() == 1) {
                // 1D to 5D
                new_shape = {1, input.size(0), 1, 1, 1};
            } else if (input.dim() == 2) {
                // 2D to 5D
                new_shape = {input.size(0), input.size(1), 1, 1, 1};
            } else if (input.dim() == 3) {
                // 3D to 5D
                new_shape = {input.size(0), input.size(1), input.size(2), 1, 1};
            } else if (input.dim() == 4) {
                // 4D to 5D
                new_shape = {input.size(0), input.size(1), input.size(2), input.size(3), 1};
            }
            
            // Reshape the tensor
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for Conv3d from the remaining data
        int64_t in_channels = input.size(1);
        
        // Parse out_channels
        int64_t out_channels = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_channels = std::abs(out_channels) % 16 + 1; // Limit to reasonable range
        }
        
        // Parse kernel_size
        int64_t kernel_d = 1, kernel_h = 1, kernel_w = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_d, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_d = std::abs(kernel_d) % 5 + 1; // Limit kernel size
        }
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_h = std::abs(kernel_h) % 5 + 1;
        }
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_w = std::abs(kernel_w) % 5 + 1;
        }
        
        // Parse stride
        int64_t stride_d = 1, stride_h = 1, stride_w = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride_d, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride_d = std::abs(stride_d) % 3 + 1;
        }
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride_h = std::abs(stride_h) % 3 + 1;
        }
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride_w = std::abs(stride_w) % 3 + 1;
        }
        
        // Parse padding
        int64_t padding_d = 0, padding_h = 0, padding_w = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_d, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_d = std::abs(padding_d) % 3;
        }
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_h = std::abs(padding_h) % 3;
        }
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_w = std::abs(padding_w) % 3;
        }
        
        // Parse dilation
        int64_t dilation_d = 1, dilation_h = 1, dilation_w = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation_d, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation_d = std::abs(dilation_d) % 2 + 1;
        }
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation_h = std::abs(dilation_h) % 2 + 1;
        }
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation_w = std::abs(dilation_w) % 2 + 1;
        }
        
        // Parse groups
        int64_t groups = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % (in_channels + 1);
            if (groups == 0) groups = 1;
            
            // Ensure in_channels is divisible by groups
            if (in_channels % groups != 0) {
                in_channels = groups; // Simplest fix
            }
        }
        
        // Parse bias flag
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 1;
        }
        
        // Create Conv3d module
        torch::nn::Conv3dOptions options(in_channels, out_channels, {kernel_d, kernel_h, kernel_w});
        options.stride({stride_d, stride_h, stride_w})
               .padding({padding_d, padding_h, padding_w})
               .dilation({dilation_d, dilation_h, dilation_w})
               .groups(groups)
               .bias(bias);
        
        torch::nn::Conv3d conv(options);
        
        // Apply Conv3d to input tensor
        torch::Tensor output = conv->forward(input);
        
        // Try different input shapes
        if (offset + 10 < Size) {
            // Create another input tensor with different shape
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Reshape to match required input format if needed
            if (input2.dim() < 5) {
                std::vector<int64_t> new_shape;
                if (input2.dim() == 0) {
                    new_shape = {1, in_channels, 1, 1, 1};
                } else if (input2.dim() == 1) {
                    new_shape = {1, in_channels, 1, 1, 1};
                } else if (input2.dim() == 2) {
                    new_shape = {input2.size(0), in_channels, 1, 1, 1};
                } else if (input2.dim() == 3) {
                    new_shape = {input2.size(0), in_channels, input2.size(2), 1, 1};
                } else if (input2.dim() == 4) {
                    new_shape = {input2.size(0), in_channels, input2.size(2), input2.size(3), 1};
                }
                
                // Only reshape if we have enough elements
                if (input2.numel() >= torch::prod(torch::tensor(new_shape)).item<int64_t>()) {
                    input2 = input2.reshape(new_shape);
                    
                    // Try to apply Conv3d to the reshaped tensor
                    try {
                        torch::Tensor output2 = conv->forward(input2);
                    } catch (const std::exception&) {
                        // Ignore exceptions from the second attempt
                    }
                }
            } else {
                // If already 5D, ensure channel dimension matches
                if (input2.size(1) != in_channels) {
                    try {
                        input2 = input2.reshape({input2.size(0), in_channels, 
                                                input2.size(2), input2.size(3), input2.size(4)});
                        torch::Tensor output2 = conv->forward(input2);
                    } catch (const std::exception&) {
                        // Ignore exceptions from the second attempt
                    }
                } else {
                    try {
                        torch::Tensor output2 = conv->forward(input2);
                    } catch (const std::exception&) {
                        // Ignore exceptions from the second attempt
                    }
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
