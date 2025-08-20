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
        
        // Extract padding values from the remaining data
        std::vector<int64_t> padding(6, 0);
        for (int i = 0; i < 6 && offset + sizeof(int64_t) <= Size; ++i) {
            int64_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative padding values to test error handling
            padding[i] = pad_value;
        }
        
        // Ensure input has at least 5D for ReplicationPad3d (batch, channels, depth, height, width)
        if (input.dim() < 5) {
            // Expand dimensions to make it 5D
            std::vector<int64_t> new_shape;
            for (int i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            while (new_shape.size() < 5) {
                new_shape.insert(new_shape.begin(), 1);
            }
            input = input.reshape(new_shape);
        }
        
        // Create ReplicationPad3d module
        torch::nn::ReplicationPad3d pad_module(padding);
        
        // Apply padding
        torch::Tensor output = pad_module->forward(input);
        
        // Verify output is not empty
        if (output.numel() == 0) {
            return 0;
        }
        
        // Try with different padding values
        if (offset + sizeof(int64_t) <= Size) {
            int64_t single_pad;
            std::memcpy(&single_pad, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Try with a single padding value
            torch::nn::ReplicationPad3d pad_module2(single_pad);
            torch::Tensor output2 = pad_module2->forward(input);
        }
        
        // Try with a tensor that has exactly 5 dimensions
        if (input.dim() > 5) {
            auto input_5d = input.flatten(0, input.dim() - 5);
            torch::Tensor output3 = pad_module->forward(input_5d);
        }
        
        // Try with different data types
        if (input.dtype() != torch::kFloat) {
            auto input_float = input.to(torch::kFloat);
            torch::Tensor output4 = pad_module->forward(input_float);
        }
        
        // Try with a tensor that has dimensions of size 0 or 1
        if (offset + sizeof(int64_t) * 5 <= Size) {
            std::vector<int64_t> edge_shape(5);
            for (int i = 0; i < 5; i++) {
                int64_t dim_size;
                std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Allow 0 or 1 as dimension sizes to test edge cases
                edge_shape[i] = std::abs(dim_size) % 2;
            }
            
            try {
                auto edge_input = torch::zeros(edge_shape);
                torch::Tensor edge_output = pad_module->forward(edge_input);
            } catch (const std::exception&) {
                // Expected exception for invalid dimensions
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