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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 5D tensor (batch_size, channels, depth, height, width)
        // If not, reshape it to 5D
        if (input.dim() < 5) {
            std::vector<int64_t> new_shape(5, 1);
            int64_t total_elements = input.numel();
            
            // Try to preserve as much of the original shape as possible
            for (int i = 0; i < std::min(5, static_cast<int>(input.dim())); i++) {
                new_shape[i] = input.size(i);
            }
            
            // Reshape tensor to 5D
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for MaxPool3d from the remaining data
        if (offset + 4 <= Size) {
            // Parse kernel size
            int kernel_size = static_cast<int>(Data[offset++]) % 5 + 1;
            
            // Parse stride
            int stride = static_cast<int>(Data[offset++]) % 5 + 1;
            
            // Parse padding
            int padding = static_cast<int>(Data[offset++]) % 3;
            
            // Parse dilation
            int dilation = static_cast<int>(Data[offset++]) % 3 + 1;
            
            // Parse ceil_mode
            bool ceil_mode = false;
            if (offset < Size) {
                ceil_mode = static_cast<bool>(Data[offset++] & 1);
            }
            
            // Create MaxPool3d module
            torch::nn::MaxPool3d max_pool(
                torch::nn::MaxPool3dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode)
            );
            
            // Apply MaxPool3d
            torch::Tensor output = max_pool->forward(input);
            
            // Optionally test other configurations if we have more data
            if (offset + 1 < Size) {
                // Try with different kernel sizes for each dimension
                int k_d = static_cast<int>(Data[offset++]) % 4 + 1;
                int k_h = static_cast<int>(Data[offset++]) % 4 + 1;
                int k_w = static_cast<int>(Data[offset++]) % 4 + 1;
                
                // Create MaxPool3d with different kernel sizes for each dimension
                torch::nn::MaxPool3d max_pool_diff_kernel(
                    torch::nn::MaxPool3dOptions({k_d, k_h, k_w})
                        .stride(stride)
                        .padding(padding)
                        .dilation(dilation)
                        .ceil_mode(ceil_mode)
                );
                
                // Apply MaxPool3d with different kernel sizes
                torch::Tensor output2 = max_pool_diff_kernel->forward(input);
            }
            
            // Try with functional max_pool3d with return_indices if we have more data
            if (offset < Size) {
                bool return_indices = static_cast<bool>(Data[offset++] & 1);
                if (return_indices) {
                    // Use functional API for return_indices
                    auto result = torch::nn::functional::max_pool3d_with_indices(
                        input,
                        torch::nn::functional::MaxPool3dFuncOptions(kernel_size)
                            .stride(stride)
                            .padding(padding)
                            .dilation(dilation)
                            .ceil_mode(ceil_mode)
                    );
                    torch::Tensor output_indices = std::get<0>(result);
                    torch::Tensor indices = std::get<1>(result);
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
