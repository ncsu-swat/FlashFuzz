#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        // Need minimum data for parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for MaxPool1d/MaxUnpool1d
        int64_t kernel_size = (Data[offset++] % 5) + 1;  // 1-5
        int64_t stride = (Data[offset++] % 5) + 1;       // 1-5
        int64_t padding = Data[offset++] % std::min(kernel_size / 2 + 1, (int64_t)3);  // Valid padding
        
        // Extract tensor dimensions
        int64_t batch_size = (Data[offset++] % 4) + 1;   // 1-4
        int64_t channels = (Data[offset++] % 8) + 1;     // 1-8
        int64_t input_length = (Data[offset++] % 20) + kernel_size;  // Ensure valid length
        
        // Determine dtype from fuzzer data
        bool use_float = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        auto dtype = use_float ? torch::kFloat32 : torch::kFloat64;
        
        // Create input tensor for MaxPool1d (3D: batch, channels, length)
        torch::Tensor original_input = torch::randn({batch_size, channels, input_length}, dtype);
        
        // If we have more data, use it to modify the tensor values
        if (offset + 4 <= Size) {
            float scale;
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(scale) && std::abs(scale) > 0.001f && std::abs(scale) < 1000.0f) {
                original_input = original_input * scale;
            }
        }
        
        // Create MaxPool1d to get valid indices
        torch::nn::MaxPool1d pool(
            torch::nn::MaxPool1dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
        );
        
        // Run MaxPool1d to get pooled output and indices
        torch::Tensor pooled_output, indices;
        try {
            auto pool_result = torch::max_pool1d_with_indices(original_input, kernel_size, stride, padding);
            pooled_output = std::get<0>(pool_result);
            indices = std::get<1>(pool_result);
        } catch (...) {
            // Shape incompatibility, silently skip
            return 0;
        }
        
        // Create MaxUnpool1d module with same parameters
        torch::nn::MaxUnpool1d unpool(
            torch::nn::MaxUnpool1dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
        );
        
        // Apply MaxUnpool1d operation
        torch::Tensor output;
        
        // Decide whether to provide output_size based on fuzzer data
        bool provide_output_size = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        try {
            if (provide_output_size) {
                // Provide the original input size as output_size
                std::vector<int64_t> output_size = {original_input.size(2)};
                output = unpool->forward(pooled_output, indices, output_size);
            } else {
                // Let it infer output size
                output = unpool->forward(pooled_output, indices);
            }
        } catch (...) {
            // Shape issues can happen, silently skip
            return 0;
        }
        
        // Verify output properties to ensure computation completed
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        // Additional operations to increase coverage
        if (offset < Size && Data[offset] % 4 == 0) {
            // Test with different input values
            torch::Tensor modified_input = pooled_output * 2.0;
            try {
                auto modified_output = unpool->forward(modified_input, indices);
                (void)modified_output.sum().item<float>();
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Force computation
        (void)output.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}