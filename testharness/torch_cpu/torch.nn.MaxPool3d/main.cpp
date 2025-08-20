#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 5D tensor (batch_size, channels, depth, height, width)
        // If not, reshape it to make it compatible with MaxPool3d
        if (input.dim() < 5) {
            std::vector<int64_t> new_shape(5, 1);
            int64_t total_elements = input.numel();
            
            // Try to preserve as much of the original shape as possible
            for (int i = 0; i < std::min(5, static_cast<int>(input.dim())); i++) {
                new_shape[i] = input.size(i);
            }
            
            // Reshape the tensor to 5D
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for MaxPool3d from the remaining data
        int64_t kernel_size = 3;
        int64_t stride = 2;
        int64_t padding = 0;
        int64_t dilation = 1;
        bool ceil_mode = false;
        
        if (offset + 5 <= Size) {
            kernel_size = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            stride = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            padding = static_cast<int64_t>(Data[offset++]) % 3;
            dilation = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            ceil_mode = Data[offset++] % 2 == 1;
        }
        
        // Create MaxPool3d module
        torch::nn::MaxPool3d max_pool(
            torch::nn::MaxPool3dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .ceil_mode(ceil_mode)
        );
        
        // Apply MaxPool3d to the input tensor
        torch::Tensor output = max_pool->forward(input);
        
        // Ensure the output is valid by accessing some elements
        if (output.numel() > 0) {
            auto accessor = output.accessor<float, 5>();
            float sum = 0.0;
            for (int i = 0; i < std::min(static_cast<int>(output.size(0)), 1); i++) {
                for (int j = 0; j < std::min(static_cast<int>(output.size(1)), 1); j++) {
                    for (int k = 0; k < std::min(static_cast<int>(output.size(2)), 1); k++) {
                        for (int l = 0; l < std::min(static_cast<int>(output.size(3)), 1); l++) {
                            for (int m = 0; m < std::min(static_cast<int>(output.size(4)), 1); m++) {
                                sum += accessor[i][j][k][l][m];
                            }
                        }
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