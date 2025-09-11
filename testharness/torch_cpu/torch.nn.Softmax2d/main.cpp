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
        // If the tensor doesn't have 4 dimensions, we'll reshape it
        if (input.dim() != 4) {
            // Get total number of elements
            int64_t total_elements = input.numel();
            
            // Create a valid 4D shape
            int64_t batch_size = 1;
            int64_t channels = 1;
            int64_t height = 1;
            int64_t width = 1;
            
            // Try to distribute elements across dimensions
            if (total_elements > 0) {
                width = std::min(total_elements, int64_t(4));
                total_elements /= width;
                
                if (total_elements > 0) {
                    height = std::min(total_elements, int64_t(4));
                    total_elements /= height;
                    
                    if (total_elements > 0) {
                        channels = std::min(total_elements, int64_t(3));
                        total_elements /= channels;
                        
                        if (total_elements > 0) {
                            batch_size = total_elements;
                        }
                    }
                }
            }
            
            // Reshape the tensor
            input = input.reshape({batch_size, channels, height, width});
        }
        
        // Create Softmax2d module
        torch::nn::Softmax2d softmax2d;
        
        // Apply Softmax2d to the input tensor
        torch::Tensor output = softmax2d->forward(input);
        
        // Try to access the output tensor to ensure computation is done
        if (output.defined()) {
            float sum = output.sum().item<float>();
            
            // Use the sum to prevent the compiler from optimizing away the computation
            if (std::isnan(sum)) {
                return 0;
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
