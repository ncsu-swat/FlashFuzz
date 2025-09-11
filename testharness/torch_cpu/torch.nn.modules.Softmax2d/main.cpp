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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Softmax2d requires a 4D tensor with shape [N, C, H, W]
        // If the tensor doesn't have 4 dimensions, reshape it
        if (input.dim() != 4) {
            // Get the total number of elements
            int64_t total_elements = input.numel();
            
            // Create a valid 4D shape
            int64_t batch_size = 1;
            int64_t channels = 1;
            int64_t height = 1;
            int64_t width = 1;
            
            // If we have enough elements, distribute them across dimensions
            if (total_elements > 0) {
                // Use the remaining data to determine dimensions if available
                if (offset + 4 <= Size) {
                    batch_size = (Data[offset] % 4) + 1;
                    offset++;
                    channels = (Data[offset] % 4) + 1;
                    offset++;
                    
                    // Calculate remaining elements for height and width
                    int64_t remaining = total_elements / (batch_size * channels);
                    
                    // Try to make height and width roughly equal
                    int64_t sqrt_remaining = static_cast<int64_t>(std::sqrt(remaining));
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
                input = input.reshape({batch_size, channels, height, width});
            } catch (const std::exception& e) {
                // If reshape fails, create a simple 4D tensor
                input = torch::ones({1, 1, 1, 1}, input.options());
            }
        }
        
        // Create Softmax2d module
        torch::nn::Softmax2d softmax2d;
        
        // Apply Softmax2d to the input tensor
        torch::Tensor output = softmax2d->forward(input);
        
        // Verify output has the same shape as input
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape doesn't match input shape");
        }
        
        // Try with different input types
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert input to the new dtype if possible
            try {
                torch::Tensor input_converted = input.to(dtype);
                torch::Tensor output_converted = softmax2d->forward(input_converted);
            } catch (const std::exception& e) {
                // Some dtypes might not be supported, that's fine
            }
        }
        
        // Try with different device if available
        try {
            if (torch::cuda::is_available()) {
                torch::Tensor input_cuda = input.cuda();
                torch::Tensor output_cuda = softmax2d->forward(input_cuda);
            }
        } catch (const std::exception& e) {
            // CUDA might not be available or might fail, that's fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
