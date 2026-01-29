#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        // Expected input shape: (N, C, L) or (C, L)
        if (input.dim() == 0) {
            input = input.unsqueeze(0).unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 1) {
            input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 2) {
            input = input.unsqueeze(0);
        }
        
        // Ensure input is float type (required for pooling)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get output size parameter from the remaining data
        int64_t output_size = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make output_size a reasonable positive value (1 to 100)
            output_size = (std::abs(output_size) % 100) + 1;
        }
        
        // Create AdaptiveMaxPool1d module
        torch::nn::AdaptiveMaxPool1d pool(output_size);
        
        // Apply the operation
        auto output = pool->forward(input);
        
        // Try with tuple output (returns indices as well)
        try {
            auto result = pool->forward_with_indices(input);
            auto pooled = std::get<0>(result);
            auto indices = std::get<1>(result);
        } catch (...) {
            // forward_with_indices might not be available or might fail
        }
        
        // Test with different input types
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            // Only test with floating point types for pooling
            torch::Dtype dtype;
            switch (dtype_selector % 3) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
                default: dtype = torch::kFloat32; break;
            }
            
            try {
                auto input_converted = input.to(dtype);
                auto output_converted = pool->forward(input_converted);
            } catch (...) {
                // Some dtype conversions might not be valid
            }
        }
        
        // Test with different batch sizes
        if (offset + sizeof(int64_t) <= Size) {
            int64_t batch_size;
            std::memcpy(&batch_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make batch_size a reasonable positive value (1 to 10)
            batch_size = (std::abs(batch_size) % 10) + 1;
            
            try {
                // Create a new input with the specified batch size
                int64_t channels = input.size(-2);
                int64_t seq_len = input.size(-1);
                auto batched_input = torch::randn({batch_size, channels, seq_len}, input.options());
                auto batched_output = pool->forward(batched_input);
            } catch (...) {
                // Some shape manipulations might not be valid
            }
        }
        
        // Test with different channel counts
        if (offset + sizeof(int64_t) <= Size) {
            int64_t channels;
            std::memcpy(&channels, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make channels a reasonable positive value (1 to 10)
            channels = (std::abs(channels) % 10) + 1;
            
            try {
                int64_t batch = input.size(0);
                int64_t seq_len = input.size(-1);
                auto channeled_input = torch::randn({batch, channels, seq_len}, input.options());
                auto channeled_output = pool->forward(channeled_input);
            } catch (...) {
                // Some shape manipulations might not be valid
            }
        }
        
        // Test with different sequence lengths
        if (offset + sizeof(int64_t) <= Size) {
            int64_t seq_len;
            std::memcpy(&seq_len, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make seq_len a reasonable positive value (1 to 100)
            seq_len = (std::abs(seq_len) % 100) + 1;
            
            try {
                int64_t batch = input.size(0);
                int64_t channels = input.size(-2);
                auto seq_input = torch::randn({batch, channels, seq_len}, input.options());
                auto seq_output = pool->forward(seq_input);
            } catch (...) {
                // Some shape manipulations might not be valid
            }
        }
        
        // Test with return_indices option
        try {
            torch::nn::AdaptiveMaxPool1dOptions opts(output_size);
            torch::nn::AdaptiveMaxPool1d pool_with_opts(opts);
            auto result_with_opts = pool_with_opts->forward(input);
        } catch (...) {
            // Options test might fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}