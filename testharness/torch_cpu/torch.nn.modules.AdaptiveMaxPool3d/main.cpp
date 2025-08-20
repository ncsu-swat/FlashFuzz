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
        
        // Ensure the input tensor has at least 5 dimensions (batch, channels, D, H, W)
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
        
        // Try to extract 3 values for D, H, W output sizes
        for (int i = 0; i < 3 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t size_val;
            std::memcpy(&size_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output size is positive (can be None/0 for adaptive)
            if (size_val < 0) {
                size_val = 0;  // Use 0 to represent None
            } else if (size_val > 100) {
                size_val = size_val % 100 + 1;  // Limit to reasonable size
            }
            
            output_size.push_back(size_val);
        }
        
        // If we couldn't extract all 3 values, fill with defaults
        while (output_size.size() < 3) {
            output_size.push_back(1);
        }
        
        // Create AdaptiveMaxPool3d module with different output size configurations
        torch::nn::AdaptiveMaxPool3d pool = nullptr;
        
        // Try different output size configurations
        if (offset < Size) {
            uint8_t config_type = Data[offset++] % 4;
            
            switch (config_type) {
                case 0:
                    // Single integer for all dimensions
                    pool = torch::nn::AdaptiveMaxPool3d(output_size[0]);
                    break;
                    
                case 1:
                    // Tuple of three integers
                    pool = torch::nn::AdaptiveMaxPool3d(
                        torch::nn::AdaptiveMaxPool3dOptions({output_size[0], output_size[1], output_size[2]}));
                    break;
                    
                case 2:
                    // Mix of dimensions (some None)
                    if (output_size[0] == 0) output_size[0] = 1;
                    pool = torch::nn::AdaptiveMaxPool3d(
                        torch::nn::AdaptiveMaxPool3dOptions({output_size[0], output_size[1], output_size[2]}));
                    break;
                    
                case 3:
                    // Default constructor (should use default output size)
                    pool = torch::nn::AdaptiveMaxPool3d(1);
                    break;
            }
        } else {
            // Default if no more data
            pool = torch::nn::AdaptiveMaxPool3d(1);
        }
        
        // Apply the pooling operation
        torch::Tensor output;
        
        // Try with return indices
        bool return_indices = false;
        if (offset < Size) {
            return_indices = Data[offset++] % 2 == 0;
        }
        
        if (return_indices) {
            torch::Tensor indices;
            std::tie(output, indices) = pool->forward_with_indices(input);
        } else {
            output = pool->forward(input);
        }
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Try to access the output tensor's properties
        auto sizes = output.sizes();
        auto dtype = output.dtype();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}