#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::min

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
        
        // Ensure we have a 5D tensor (batch, channels, depth, height, width)
        // If not, reshape it to 5D
        if (input.dim() < 5) {
            std::vector<int64_t> new_shape(5, 1);
            int64_t total_elements = input.numel();
            
            // Try to distribute elements across dimensions
            for (int i = 0; i < std::min(static_cast<int>(input.dim()), 5); i++) {
                if (i < input.dim()) {
                    new_shape[i] = input.size(i);
                }
            }
            
            input = input.reshape(new_shape);
        }
        
        // Parse output size parameters from the remaining data
        std::vector<int64_t> output_size;
        
        // We need 1-3 values for output_size
        int num_output_dims = 1;
        if (offset + 1 < Size) {
            num_output_dims = (Data[offset++] % 3) + 1; // 1, 2, or 3 dimensions
        }
        
        for (int i = 0; i < num_output_dims; i++) {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t dim_value;
                std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Ensure output size is positive
                dim_value = std::abs(dim_value) % 10 + 1;
                output_size.push_back(dim_value);
            } else {
                // Default value if not enough data
                output_size.push_back(1);
            }
        }
        
        // Create AdaptiveMaxPool3d module
        torch::nn::AdaptiveMaxPool3d pool = nullptr;
        
        // Set output size based on the number of dimensions
        if (num_output_dims == 1) {
            pool = torch::nn::AdaptiveMaxPool3d(torch::nn::AdaptiveMaxPool3dOptions(output_size[0]));
        } else if (num_output_dims == 2) {
            pool = torch::nn::AdaptiveMaxPool3d(torch::nn::AdaptiveMaxPool3dOptions(std::vector<int64_t>{output_size[0], output_size[1]}));
        } else {
            pool = torch::nn::AdaptiveMaxPool3d(torch::nn::AdaptiveMaxPool3dOptions(std::vector<int64_t>{output_size[0], output_size[1], output_size[2]}));
        }
        
        // Apply the pooling operation
        auto output = pool->forward(input);
        
        // Try to access the indices if available
        if (offset < Size && Data[offset] % 2 == 0) {
            auto result_tuple = pool->forward_with_indices(input);
            auto result = std::get<0>(result_tuple);
            auto indices = std::get<1>(result_tuple);
            
            // Use the indices to ensure they're computed
            auto indices_sum = indices.sum().item<float>();
            if (indices_sum < 0) {
                // This is just to use the value and prevent optimization
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