#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 5D tensor (batch_size, channels, depth, height, width)
        // If not, reshape it to 5D
        if (input.dim() < 5) {
            std::vector<int64_t> new_shape;
            
            // Keep original dimensions
            for (int i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            
            // Add missing dimensions
            while (new_shape.size() < 5) {
                new_shape.push_back(1);
            }
            
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for AvgPool3d from the remaining data
        int64_t kernel_size = 1;
        int64_t stride = 1;
        int64_t padding = 0;
        bool ceil_mode = false;
        bool count_include_pad = true;
        int64_t divisor_override = 0;
        
        if (offset + 1 < Size) {
            kernel_size = (Data[offset++] % 5) + 1; // 1-5
        }
        
        if (offset + 1 < Size) {
            stride = (Data[offset++] % 3) + 1; // 1-3
        }
        
        if (offset + 1 < Size) {
            padding = Data[offset++] % 3; // 0-2
        }
        
        if (offset + 1 < Size) {
            ceil_mode = Data[offset++] % 2; // 0-1 (false-true)
        }
        
        if (offset + 1 < Size) {
            count_include_pad = Data[offset++] % 2; // 0-1 (false-true)
        }
        
        if (offset + 1 < Size) {
            // divisor_override should be 0 (None) or a positive integer
            divisor_override = Data[offset++] % 4; // 0-3
        }
        
        // Create AvgPool3d module with various parameters
        torch::nn::AvgPool3d avg_pool = nullptr;
        
        // Try different parameter combinations
        if (offset % 4 == 0) {
            // Single integer for kernel_size, stride, padding
            avg_pool = torch::nn::AvgPool3d(
                torch::nn::AvgPool3dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .ceil_mode(ceil_mode)
                    .count_include_pad(count_include_pad)
            );
        } else if (offset % 4 == 1) {
            // Try with tuple for kernel_size
            avg_pool = torch::nn::AvgPool3d(
                torch::nn::AvgPool3dOptions({kernel_size, kernel_size, kernel_size})
                    .stride(stride)
                    .padding(padding)
                    .ceil_mode(ceil_mode)
                    .count_include_pad(count_include_pad)
            );
        } else if (offset % 4 == 2) {
            // Try with tuple for stride
            avg_pool = torch::nn::AvgPool3d(
                torch::nn::AvgPool3dOptions(kernel_size)
                    .stride({stride, stride, stride})
                    .padding(padding)
                    .ceil_mode(ceil_mode)
                    .count_include_pad(count_include_pad)
            );
        } else {
            // Try with tuple for padding
            c10::optional<int64_t> divisor_opt = divisor_override > 0 ? c10::optional<int64_t>(divisor_override) : c10::nullopt;
            avg_pool = torch::nn::AvgPool3d(
                torch::nn::AvgPool3dOptions(kernel_size)
                    .stride(stride)
                    .padding({padding, padding, padding})
                    .ceil_mode(ceil_mode)
                    .count_include_pad(count_include_pad)
                    .divisor_override(divisor_opt)
            );
        }
        
        // Apply the AvgPool3d operation
        torch::Tensor output = avg_pool->forward(input);
        
        // Try to access output properties to ensure computation completed
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}