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
        
        // Ensure the input tensor has at least 3 dimensions for AdaptiveAvgPool1d
        // (batch_size, channels, input_size)
        if (input.dim() < 1) {
            input = input.unsqueeze(0);
        }
        
        // Extract output size from the remaining data
        int64_t output_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_size is within reasonable bounds
            output_size = std::abs(output_size) % 100 + 1;
        }
        
        // Create the AdaptiveAvgPool1d module
        torch::nn::AdaptiveAvgPool1d pool(output_size);
        
        // Apply the pooling operation
        torch::Tensor output = pool(input);
        
        // Verify the output has the expected size in the last dimension
        if (output.dim() > 0 && output.size(-1) != output_size) {
            throw std::runtime_error("Output size mismatch");
        }
        
        // Try with different output sizes
        if (offset + sizeof(int64_t) <= Size) {
            int64_t output_size2;
            std::memcpy(&output_size2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_size is within reasonable bounds
            output_size2 = std::abs(output_size2) % 100 + 1;
            
            // Create another AdaptiveAvgPool1d module with different output size
            torch::nn::AdaptiveAvgPool1d pool2(output_size2);
            
            // Apply the pooling operation
            torch::Tensor output2 = pool2(input);
            
            // Verify the output has the expected size in the last dimension
            if (output2.dim() > 0 && output2.size(-1) != output_size2) {
                throw std::runtime_error("Output size mismatch for second pooling");
            }
        }
        
        // Try with a vector of output sizes (for multi-dimensional output)
        if (offset + sizeof(int64_t) <= Size) {
            int64_t output_size_vec_raw;
            std::memcpy(&output_size_vec_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_size is within reasonable bounds
            output_size_vec_raw = std::abs(output_size_vec_raw) % 100 + 1;
            
            // Create a vector with a single output size
            std::vector<int64_t> output_size_vec = {output_size_vec_raw};
            
            // Create another AdaptiveAvgPool1d module with vector output size
            torch::nn::AdaptiveAvgPool1d pool3(output_size_vec);
            
            // Apply the pooling operation
            torch::Tensor output3 = pool3(input);
            
            // Verify the output has the expected size in the last dimension
            if (output3.dim() > 0 && output3.size(-1) != output_size_vec_raw) {
                throw std::runtime_error("Output size mismatch for vector-specified pooling");
            }
        }
        
        // Try with a None/0 output size (should preserve the input size)
        if (offset < Size) {
            // Create an AdaptiveAvgPool1d module with None output size
            torch::nn::AdaptiveAvgPool1d pool_none(0);
            
            // Apply the pooling operation
            torch::Tensor output_none = pool_none(input);
            
            // For None output size, the output should have the same size as input in the last dimension
            if (input.dim() > 0 && output_none.dim() > 0 && 
                output_none.size(-1) != input.size(-1)) {
                throw std::runtime_error("Output size mismatch for None output_size");
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