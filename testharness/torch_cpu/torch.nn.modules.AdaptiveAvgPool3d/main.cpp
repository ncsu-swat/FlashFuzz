#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor has at least 5 dimensions (batch, channels, D, H, W)
        // If not, reshape it to have 5 dimensions
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
        
        // Parse output size parameters from the remaining data
        std::vector<int64_t> output_size;
        
        // Try to get 3 values for D, H, W output sizes
        for (int i = 0; i < 3; i++) {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t size_val;
                std::memcpy(&size_val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Make sure output size is reasonable (can be None/0 or positive)
                if (size_val < 0) {
                    size_val = 0;  // Treat negative as None
                } else if (size_val > 100) {
                    size_val = size_val % 100 + 1;  // Limit to reasonable size
                }
                
                output_size.push_back(size_val);
            } else {
                // Default to 1 if not enough data
                output_size.push_back(1);
            }
        }
        
        // Create AdaptiveAvgPool3d module with different output size configurations
        torch::nn::AdaptiveAvgPool3d pool = nullptr;
        
        // Try different output size configurations
        if (offset < Size) {
            uint8_t config_type = Data[offset++] % 4;
            
            switch (config_type) {
                case 0:
                    // Single integer for all dimensions
                    pool = torch::nn::AdaptiveAvgPool3d(output_size[0]);
                    break;
                    
                case 1:
                    // Tuple of 3 values
                    pool = torch::nn::AdaptiveAvgPool3d(torch::nn::AdaptiveAvgPool3dOptions(
                        {output_size[0], output_size[1], output_size[2]}));
                    break;
                    
                case 2:
                    // Some dimensions as 0/None (meaning preserve input size)
                    if (output_size[0] == 0) output_size[0] = 1;
                    if (output_size[1] == 0) output_size[1] = 1;
                    if (output_size[2] == 0) output_size[2] = 1;
                    pool = torch::nn::AdaptiveAvgPool3d(torch::nn::AdaptiveAvgPool3dOptions(
                        {output_size[0], output_size[1], output_size[2]}));
                    break;
                    
                case 3:
                    // Default constructor (should preserve input size)
                    pool = torch::nn::AdaptiveAvgPool3d(torch::nn::AdaptiveAvgPool3dOptions(1));
                    break;
            }
        } else {
            // Default if not enough data
            pool = torch::nn::AdaptiveAvgPool3d(torch::nn::AdaptiveAvgPool3dOptions(1));
        }
        
        // Apply the pooling operation
        torch::Tensor output = pool->forward(input);
        
        // Optionally test other edge cases
        if (offset < Size && Data[offset] % 2 == 0) {
            // Test with empty batch dimension
            if (input.size(0) > 0) {
                torch::Tensor empty_input = input.slice(0, 0, 0);
                torch::Tensor empty_output = pool->forward(empty_input);
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