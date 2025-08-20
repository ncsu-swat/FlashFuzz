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
        
        // Ensure input has at least 3 dimensions for AdaptiveMaxPool1d
        // If not, add dimensions as needed
        if (input.dim() == 0) {
            input = input.unsqueeze(0).unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 1) {
            input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 2) {
            input = input.unsqueeze(0);
        }
        
        // Get output size parameter from the remaining data
        int64_t output_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make output_size a reasonable value (can be negative to test error cases)
            output_size = output_size % 100;
        }
        
        // Create AdaptiveMaxPool1d module
        torch::nn::AdaptiveMaxPool1d pool(output_size);
        
        // Apply the operation
        auto output = pool->forward(input);
        
        // Try with tuple output (returns indices as well)
        auto [pooled, indices] = pool->forward_with_indices(input);
        
        // Test with different input types
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert input to the new dtype if possible
            try {
                auto input_converted = input.to(dtype);
                auto output_converted = pool->forward(input_converted);
            } catch (const std::exception& e) {
                // Some dtype conversions might not be valid, that's fine
            }
        }
        
        // Test with different batch sizes
        if (offset + sizeof(int64_t) <= Size) {
            int64_t batch_size;
            std::memcpy(&batch_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make batch_size a reasonable value
            batch_size = std::abs(batch_size) % 10 + 1;
            
            // Create a new input with the specified batch size
            std::vector<int64_t> new_shape = {batch_size};
            for (int64_t i = 1; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            
            try {
                auto batched_input = input.expand(new_shape);
                auto batched_output = pool->forward(batched_input);
            } catch (const std::exception& e) {
                // Some shape manipulations might not be valid, that's fine
            }
        }
        
        // Test with different channel counts
        if (offset + sizeof(int64_t) <= Size) {
            int64_t channels;
            std::memcpy(&channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make channels a reasonable value
            channels = std::abs(channels) % 10 + 1;
            
            try {
                // Create a new input with the specified number of channels
                std::vector<int64_t> channel_shape = {input.size(0), channels};
                for (int64_t i = 2; i < input.dim(); i++) {
                    channel_shape.push_back(input.size(i));
                }
                
                auto channeled_input = torch::ones(channel_shape, input.options());
                auto channeled_output = pool->forward(channeled_input);
            } catch (const std::exception& e) {
                // Some shape manipulations might not be valid, that's fine
            }
        }
        
        // Test with different sequence lengths
        if (offset + sizeof(int64_t) <= Size) {
            int64_t seq_len;
            std::memcpy(&seq_len, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make seq_len a reasonable value (can be negative to test error cases)
            seq_len = seq_len % 100;
            
            try {
                // Create a new input with the specified sequence length
                std::vector<int64_t> seq_shape = {input.size(0), input.size(1), seq_len};
                
                auto seq_input = torch::ones(seq_shape, input.options());
                auto seq_output = pool->forward(seq_input);
            } catch (const std::exception& e) {
                // Some shape manipulations might not be valid, that's fine
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