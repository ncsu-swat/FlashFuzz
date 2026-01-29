#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need enough data for parameters
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for MaxUnpool1d
        int64_t kernel_size = (Data[offset++] % 5) + 1; // 1-5
        int64_t stride = (Data[offset++] % 5) + 1; // 1-5
        int64_t padding = Data[offset++] % 3; // 0-2
        
        // Extract tensor dimensions
        int64_t batch_size = (Data[offset++] % 4) + 1; // 1-4
        int64_t channels = (Data[offset++] % 8) + 1; // 1-8
        int64_t length = (Data[offset++] % 16) + 1; // 1-16
        
        // Determine if we should provide output_size
        bool use_output_size = Data[offset++] % 2;
        
        // Create input tensor with proper shape for MaxUnpool1d: (N, C, L)
        torch::Tensor input = torch::randn({batch_size, channels, length}, torch::kFloat32);
        
        // Populate input from fuzzer data if available
        if (offset + input.numel() * sizeof(float) <= Size) {
            float* input_data = input.data_ptr<float>();
            for (int64_t i = 0; i < input.numel() && offset + sizeof(float) <= Size; i++) {
                std::memcpy(&input_data[i], Data + offset, sizeof(float));
                offset += sizeof(float);
            }
        }
        
        // Create indices tensor with same shape as input
        // Indices must be in valid range for the unpooled output
        torch::Tensor indices = torch::zeros({batch_size, channels, length}, torch::kInt64);
        
        // Fill indices with valid values (each index should be in [0, kernel_size) offset by position * stride)
        int64_t* idx_data = indices.data_ptr<int64_t>();
        for (int64_t i = 0; i < indices.numel(); i++) {
            int64_t pos = i % length; // position in the length dimension
            int64_t base_idx = pos * stride;
            int64_t local_idx = 0;
            if (offset < Size) {
                local_idx = Data[offset++] % kernel_size;
            }
            idx_data[i] = base_idx + local_idx;
        }
        
        // Create MaxUnpool1d options
        torch::nn::MaxUnpool1dOptions options(kernel_size);
        options.stride(stride);
        options.padding(padding);
        
        // Create MaxUnpool1d module
        torch::nn::MaxUnpool1d unpool(options);
        
        // Calculate expected output size
        int64_t output_length = (length - 1) * stride - 2 * padding + kernel_size;
        
        try {
            torch::Tensor output;
            
            if (use_output_size && output_length > 0) {
                // Optionally provide explicit output_size
                std::vector<int64_t> out_size = {batch_size, channels, output_length};
                output = unpool->forward(input, indices, out_size);
            } else {
                output = unpool->forward(input, indices);
            }
            
            // Access output to ensure computation
            volatile auto sum = output.sum().item<float>();
            (void)sum;
        }
        catch (const c10::Error&) {
            // Shape/index errors are expected with random data
        }
        catch (const std::runtime_error&) {
            // Runtime errors from invalid configurations
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}