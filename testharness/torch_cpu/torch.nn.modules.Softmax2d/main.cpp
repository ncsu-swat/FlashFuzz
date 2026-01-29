#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

// --- Fuzzer Entry Point ---
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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Softmax2d requires a 4D tensor with shape [N, C, H, W]
        // If the tensor doesn't have 4 dimensions, reshape it
        if (input.dim() != 4) {
            // Get the total number of elements
            int64_t total_elements = input.numel();
            
            if (total_elements == 0) {
                return 0;
            }
            
            // Create a valid 4D shape
            int64_t batch_size = 1;
            int64_t channels = 1;
            int64_t height = 1;
            int64_t width = 1;
            
            // Use the remaining data to determine dimensions if available
            if (offset + 2 <= Size) {
                batch_size = (Data[offset] % 4) + 1;
                offset++;
                channels = (Data[offset] % 4) + 1;
                offset++;
                
                // Calculate remaining elements for height and width
                int64_t remaining = total_elements / (batch_size * channels);
                
                if (remaining > 0) {
                    // Try to make height and width roughly equal
                    int64_t sqrt_remaining = static_cast<int64_t>(std::sqrt(static_cast<double>(remaining)));
                    height = std::max<int64_t>(1, sqrt_remaining);
                    width = std::max<int64_t>(1, remaining / height);
                    
                    // Adjust to ensure total elements match
                    while (batch_size * channels * height * width > total_elements) {
                        if (width > 1) width--;
                        else if (height > 1) height--;
                        else if (channels > 1) channels--;
                        else if (batch_size > 1) batch_size--;
                        else break;
                    }
                }
            }
            
            // Reshape the tensor to [batch_size, channels, height, width]
            try {
                int64_t needed_elements = batch_size * channels * height * width;
                if (needed_elements <= total_elements && needed_elements > 0) {
                    input = input.flatten().slice(0, 0, needed_elements).reshape({batch_size, channels, height, width});
                } else {
                    // Create a simple 4D tensor from available elements
                    input = input.flatten().slice(0, 0, std::min<int64_t>(total_elements, 1)).reshape({1, 1, 1, -1});
                    if (input.size(3) == 0) {
                        input = torch::ones({1, 1, 1, 1}, input.options());
                    }
                }
            } catch (...) {
                // If reshape fails, create a simple 4D tensor
                input = torch::ones({1, 1, 1, 1}, input.options());
            }
        }
        
        // Ensure input is floating point for softmax
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create Softmax2d module
        torch::nn::Softmax2d softmax2d;
        
        // Apply Softmax2d to the input tensor
        torch::Tensor output = softmax2d(input);
        
        // Verify output has the same shape as input
        if (output.sizes() != input.sizes()) {
            std::cerr << "Output shape doesn't match input shape" << std::endl;
        }
        
        // Try with different input types
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Only try floating point types as softmax requires them
            if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || dtype == torch::kFloat16) {
                try {
                    torch::Tensor input_converted = input.to(dtype);
                    torch::Tensor output_converted = softmax2d(input_converted);
                } catch (...) {
                    // Some dtypes might not be supported, silently continue
                }
            }
        }
        
        // Test with different batch sizes and channel counts if we have more data
        if (offset + 4 <= Size) {
            int64_t new_batch = (Data[offset++] % 3) + 1;
            int64_t new_channels = (Data[offset++] % 4) + 1;
            int64_t new_height = (Data[offset++] % 8) + 1;
            int64_t new_width = (Data[offset++] % 8) + 1;
            
            try {
                torch::Tensor new_input = torch::randn({new_batch, new_channels, new_height, new_width});
                torch::Tensor new_output = softmax2d(new_input);
            } catch (...) {
                // Silently handle any failures
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