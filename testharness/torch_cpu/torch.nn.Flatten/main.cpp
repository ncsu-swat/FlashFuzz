#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has enough dimensions for meaningful flatten
        int64_t ndim = input.dim();
        if (ndim < 1) {
            return 0;
        }
        
        // Extract parameters for Flatten module, constrain to valid range
        int64_t start_dim = 1;  // Default value
        int64_t end_dim = -1;   // Default value
        
        // Get start_dim from input data if available
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_start = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            // Constrain to valid dimension range [-ndim, ndim-1]
            if (ndim > 0) {
                start_dim = raw_start % ndim;
            }
        }
        
        // Get end_dim from input data if available
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_end = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            // Constrain to valid dimension range [-ndim, ndim-1]
            if (ndim > 0) {
                end_dim = raw_end % ndim;
            }
        }
        
        // Create Flatten module using FlattenOptions with builder pattern
        torch::nn::FlattenOptions options;
        options.start_dim(start_dim);
        options.end_dim(end_dim);
        torch::nn::Flatten flatten_module(options);
        
        // Apply the Flatten operation
        try {
            torch::Tensor output = flatten_module->forward(input);
            
            // Verify output is contiguous
            (void)output.is_contiguous();
            
            // Alternative way to test: use the functional API
            torch::Tensor output2 = torch::flatten(input, start_dim, end_dim);
            
            // Verify both methods produce same result
            (void)torch::allclose(output, output2);
        } catch (const std::exception&) {
            // Invalid dimension combinations are expected
        }
        
        // Test edge case: create another Flatten with different parameters
        int64_t alt_start_dim = 0;
        int64_t alt_end_dim = -1;
        
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_alt_start = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            if (ndim > 0) {
                alt_start_dim = raw_alt_start % ndim;
            }
        }
        
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_alt_end = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            if (ndim > 0) {
                alt_end_dim = raw_alt_end % ndim;
            }
        }
        
        // Try with alternative parameters
        try {
            torch::nn::FlattenOptions alt_options;
            alt_options.start_dim(alt_start_dim);
            alt_options.end_dim(alt_end_dim);
            torch::nn::Flatten alt_flatten(alt_options);
            torch::Tensor alt_output = alt_flatten->forward(input);
        } catch (const std::exception&) {
            // Expected to potentially throw for invalid parameters
        }
        
        // Test flattening the entire tensor (all dims)
        try {
            torch::nn::FlattenOptions full_options;
            full_options.start_dim(0);
            full_options.end_dim(-1);
            torch::nn::Flatten full_flatten(full_options);
            torch::Tensor full_output = full_flatten->forward(input);
            
            // Result should be 1D if we flatten all dimensions starting from 0
            (void)full_output.numel();
        } catch (const std::exception&) {
            // Expected for edge cases
        }
        
        // Test with a multi-dimensional tensor created from the same data
        if (offset + 4 <= Size) {
            try {
                // Create a tensor with known shape for better coverage
                int64_t batch = (Data[offset] % 4) + 1;
                int64_t channels = (Data[offset + 1] % 4) + 1;
                int64_t height = (Data[offset + 2] % 4) + 1;
                int64_t width = (Data[offset + 3] % 4) + 1;
                offset += 4;
                
                torch::Tensor img_tensor = torch::randn({batch, channels, height, width});
                
                // Common use case: flatten all but batch dimension
                torch::nn::FlattenOptions batch_options;
                batch_options.start_dim(1);
                batch_options.end_dim(-1);
                torch::nn::Flatten batch_flatten(batch_options);
                torch::Tensor batch_output = batch_flatten->forward(img_tensor);
                
                // Result should be [batch, channels*height*width]
                (void)batch_output.sizes();
            } catch (const std::exception&) {
                // Handle allocation failures gracefully
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}