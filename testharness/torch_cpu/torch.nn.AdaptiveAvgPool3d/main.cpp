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
        
        // Try to extract 3 values for D, H, W output sizes
        for (int i = 0; i < 3 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t size_val;
            std::memcpy(&size_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure output size is positive (can be None/0 for adaptive)
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
        
        // Create the AdaptiveAvgPool3d module
        torch::nn::AdaptiveAvgPool3d pool(output_size);
        
        // Apply the pooling operation
        torch::Tensor output = pool->forward(input);
        
        // Optional: Check that output has the expected shape
        auto output_shape = output.sizes();
        if (output_shape.size() != input.dim()) {
            throw std::runtime_error("Output tensor has wrong number of dimensions");
        }
        
        // Batch and channel dimensions should be preserved
        if (output_shape[0] != input.size(0) || output_shape[1] != input.size(1)) {
            throw std::runtime_error("Batch or channel dimensions changed unexpectedly");
        }
        
        // Spatial dimensions should match our requested output size
        for (int i = 0; i < 3; i++) {
            if (output_size[i] != 0 && output_shape[i+2] != output_size[i]) {
                throw std::runtime_error("Output spatial dimensions don't match requested size");
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