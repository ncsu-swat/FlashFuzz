#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 6 bytes left for padding values
        if (Size - offset < 6) {
            return 0;
        }
        
        // CircularPad3d expects a 4D or 5D tensor
        // 4D: (channels, depth, height, width)
        // 5D: (batch, channels, depth, height, width)
        if (input_tensor.dim() < 4) {
            // Expand to 5D
            std::vector<int64_t> new_shape;
            for (int i = 0; i < 5 - input_tensor.dim(); i++) {
                new_shape.push_back(1);
            }
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                new_shape.push_back(input_tensor.size(i));
            }
            input_tensor = input_tensor.reshape(new_shape);
        } else if (input_tensor.dim() > 5) {
            // Collapse to 5D
            std::vector<int64_t> new_shape;
            int64_t collapsed_size = 1;
            for (int64_t i = 0; i < input_tensor.dim() - 4; i++) {
                collapsed_size *= input_tensor.size(i);
            }
            new_shape.push_back(collapsed_size);
            for (int64_t i = input_tensor.dim() - 4; i < input_tensor.dim(); i++) {
                new_shape.push_back(input_tensor.size(i));
            }
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Get spatial dimensions for padding constraints
        int64_t depth_dim = input_tensor.dim() == 5 ? 2 : 1;
        int64_t height_dim = input_tensor.dim() == 5 ? 3 : 2;
        int64_t width_dim = input_tensor.dim() == 5 ? 4 : 3;
        
        int64_t depth = input_tensor.size(depth_dim);
        int64_t height = input_tensor.size(height_dim);
        int64_t width = input_tensor.size(width_dim);
        
        // For circular padding, padding must be less than the corresponding dimension
        // Extract padding values from the input data (left, right, top, bottom, front, back)
        std::vector<int64_t> padding(6);
        
        // Padding for width (left, right)
        padding[0] = (width > 0) ? static_cast<int64_t>(Data[offset++]) % width : 0;
        padding[1] = (width > 0) ? static_cast<int64_t>(Data[offset++]) % width : 0;
        
        // Padding for height (top, bottom)
        padding[2] = (height > 0) ? static_cast<int64_t>(Data[offset++]) % height : 0;
        padding[3] = (height > 0) ? static_cast<int64_t>(Data[offset++]) % height : 0;
        
        // Padding for depth (front, back)
        padding[4] = (depth > 0) ? static_cast<int64_t>(Data[offset++]) % depth : 0;
        padding[5] = (depth > 0) ? static_cast<int64_t>(Data[offset++]) % depth : 0;
        
        // Apply the circular padding operation using functional interface
        // This is the equivalent of torch.nn.CircularPad3d in Python
        torch::Tensor output = torch::nn::functional::pad(
            input_tensor, 
            torch::nn::functional::PadFuncOptions(padding).mode(torch::kCircular)
        );
        
        // Basic sanity check without assertions that would crash
        if (output.defined()) {
            // Access the data to ensure computation happens
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test with different padding configurations
        // Try with only 4 padding values (2D padding on 3D spatial tensor)
        if (Size - offset >= 4) {
            try {
                std::vector<int64_t> padding_4 = {
                    (width > 0) ? static_cast<int64_t>(Data[offset++]) % width : 0,
                    (width > 0) ? static_cast<int64_t>(Data[offset++]) % width : 0,
                    (height > 0) ? static_cast<int64_t>(Data[offset++]) % height : 0,
                    (height > 0) ? static_cast<int64_t>(Data[offset++]) % height : 0
                };
                
                torch::Tensor output_4 = torch::nn::functional::pad(
                    input_tensor,
                    torch::nn::functional::PadFuncOptions(padding_4).mode(torch::kCircular)
                );
                
                if (output_4.defined()) {
                    volatile float sum = output_4.sum().item<float>();
                    (void)sum;
                }
            } catch (const std::exception &) {
                // Some configurations may fail - that's expected
            }
        }
        
        // Test with 2 padding values (1D padding)
        if (Size - offset >= 2) {
            try {
                std::vector<int64_t> padding_2 = {
                    (width > 0) ? static_cast<int64_t>(Data[offset++]) % width : 0,
                    (width > 0) ? static_cast<int64_t>(Data[offset++]) % width : 0
                };
                
                torch::Tensor output_2 = torch::nn::functional::pad(
                    input_tensor,
                    torch::nn::functional::PadFuncOptions(padding_2).mode(torch::kCircular)
                );
                
                if (output_2.defined()) {
                    volatile float sum = output_2.sum().item<float>();
                    (void)sum;
                }
            } catch (const std::exception &) {
                // Some configurations may fail - that's expected
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