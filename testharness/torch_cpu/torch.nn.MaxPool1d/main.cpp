#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least enough bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        // Extract parameters for MaxPool1d from the data
        // Extract kernel_size
        int32_t kernel_size;
        std::memcpy(&kernel_size, Data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        kernel_size = std::abs(kernel_size) % 16 + 1; // 1-16
        
        // Extract stride
        int32_t stride;
        std::memcpy(&stride, Data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        stride = std::abs(stride) % 16 + 1; // 1-16
        
        // Extract padding
        int32_t padding;
        std::memcpy(&padding, Data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        padding = std::abs(padding) % (kernel_size / 2 + 1); // padding <= kernel_size / 2
        
        // Extract dilation
        int32_t dilation;
        std::memcpy(&dilation, Data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        dilation = std::abs(dilation) % 4 + 1; // 1-4
        
        // Extract ceil_mode and return_indices
        bool ceil_mode = false;
        bool return_indices = false;
        if (offset < Size) {
            ceil_mode = Data[offset] & 0x1;
            return_indices = Data[offset] & 0x2;
            offset++;
        }
        
        // Extract tensor dimensions for proper 3D shape (N, C, L)
        int64_t batch_size = 1;
        int64_t channels = 1;
        int64_t length = kernel_size + 1; // Minimum valid length
        
        if (offset + 3 <= Size) {
            batch_size = (Data[offset] % 8) + 1;     // 1-8
            channels = (Data[offset + 1] % 16) + 1;  // 1-16
            // Length must be >= (kernel_size - 1) * dilation + 1 after padding
            int64_t min_length = (kernel_size - 1) * dilation + 1;
            length = min_length + (Data[offset + 2] % 64); // min_length to min_length+63
            offset += 3;
        }
        
        // Create 3D input tensor with proper shape for MaxPool1d
        torch::Tensor input;
        if (offset < Size && Data[offset - 1] & 0x4) {
            // Sometimes use 2D unbatched input (C, L)
            input = torch::randn({channels, length});
        } else {
            // Use 3D batched input (N, C, L)
            input = torch::randn({batch_size, channels, length});
        }
        
        // Seed the random values from fuzzer data if available
        if (offset + 4 <= Size) {
            int32_t seed;
            std::memcpy(&seed, Data + offset, sizeof(int32_t));
            torch::manual_seed(seed);
            input = torch::randn_like(input);
        }
        
        // Inner try-catch for expected shape/parameter validation failures
        try
        {
            // Create MaxPool1d module
            auto options = torch::nn::MaxPool1dOptions(kernel_size)
                              .stride(stride)
                              .padding(padding)
                              .dilation(dilation)
                              .ceil_mode(ceil_mode);
            
            torch::nn::MaxPool1d pool(options);
            
            if (return_indices) {
                // Test forward_with_indices
                auto result = pool->forward_with_indices(input);
                std::get<0>(result).sum().item<float>();
                // Verify indices tensor exists
                std::get<1>(result).numel();
            } else {
                // Test regular forward
                torch::Tensor output = pool->forward(input);
                // Ensure the output is materialized
                output.sum().item<float>();
            }
        }
        catch (const c10::Error &e)
        {
            // Expected errors from invalid parameter combinations - silently ignore
        }
        catch (const std::runtime_error &e)
        {
            // Expected runtime errors from shape mismatches - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}