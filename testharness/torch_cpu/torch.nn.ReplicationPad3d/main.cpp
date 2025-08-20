#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the input tensor and padding values
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 5D tensor (batch, channels, depth, height, width)
        // ReplicationPad3d expects 5D input with at least 3 spatial dimensions
        if (input.dim() < 5) {
            // Pad dimensions to make it 5D
            std::vector<int64_t> new_shape;
            for (int i = 0; i < 5 - input.dim(); i++) {
                new_shape.push_back(1);
            }
            for (int i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            input = input.reshape(new_shape);
        }
        
        // Extract padding values from the remaining data
        std::vector<int64_t> padding(6, 0);
        for (int i = 0; i < 6 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Limit padding size to avoid excessive memory usage
            padding[i] = std::abs(pad_value) % 10;
        }
        
        // Create ReplicationPad3d module
        torch::nn::ReplicationPad3d pad_module(torch::nn::ReplicationPad3dOptions(
            {padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]}
        ));
        
        // Apply padding
        torch::Tensor output = pad_module->forward(input);
        
        // Try with different padding values
        if (offset + sizeof(int64_t) <= Size) {
            int64_t alt_pad_value;
            std::memcpy(&alt_pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Use a single padding value for all dimensions
            int64_t single_pad = std::abs(alt_pad_value) % 5;
            torch::nn::ReplicationPad3d alt_pad_module((torch::nn::ReplicationPad3dOptions(single_pad)));
            torch::Tensor alt_output = alt_pad_module->forward(input);
        }
        
        // Try with negative padding values (should trigger exceptions)
        if (offset + sizeof(int64_t) <= Size) {
            int64_t neg_pad_value;
            std::memcpy(&neg_pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Use negative padding value
            int64_t neg_pad = -1 * (std::abs(neg_pad_value) % 10 + 1);
            try {
                torch::nn::ReplicationPad3d neg_pad_module(torch::nn::ReplicationPad3dOptions(
                    {neg_pad, 1, 1, 1, 1, 1}
                ));
                torch::Tensor neg_output = neg_pad_module->forward(input);
            } catch (...) {
                // Expected exception for negative padding
            }
        }
        
        // Try with padding larger than input dimensions
        if (offset + sizeof(int64_t) <= Size) {
            int64_t large_pad_value;
            std::memcpy(&large_pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Use large padding value
            int64_t large_pad = 20 + (std::abs(large_pad_value) % 100);
            try {
                torch::nn::ReplicationPad3d large_pad_module(torch::nn::ReplicationPad3dOptions(
                    {large_pad, large_pad, large_pad, large_pad, large_pad, large_pad}
                ));
                torch::Tensor large_output = large_pad_module->forward(input);
            } catch (...) {
                // May throw if padding is too large
            }
        }
        
        // Try with asymmetric padding
        if (offset + 6*sizeof(int64_t) <= Size) {
            std::vector<int64_t> asym_padding(6, 0);
            for (int i = 0; i < 6; i++) {
                int64_t pad_value;
                std::memcpy(&pad_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                asym_padding[i] = std::abs(pad_value) % 5;
            }
            
            torch::nn::ReplicationPad3d asym_pad_module(torch::nn::ReplicationPad3dOptions(
                {asym_padding[0], asym_padding[1], asym_padding[2], 
                 asym_padding[3], asym_padding[4], asym_padding[5]}
            ));
            torch::Tensor asym_output = asym_pad_module->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}