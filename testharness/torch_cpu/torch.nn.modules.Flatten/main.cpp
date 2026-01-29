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
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get tensor dimensions for valid parameter ranges
        int64_t ndim = input.dim();
        if (ndim == 0) {
            // Scalar tensors can't really be flattened meaningfully
            return 0;
        }
        
        // Extract parameters for Flatten module from the remaining data
        int64_t start_dim = 1;  // Default value
        int64_t end_dim = -1;   // Default value
        
        // If we have more data, use it to set start_dim within valid range
        if (offset + 1 <= Size) {
            int8_t raw_val = static_cast<int8_t>(Data[offset]);
            offset += 1;
            // Map to range [-ndim, ndim-1] for valid PyTorch indexing
            if (ndim > 0) {
                start_dim = raw_val % ndim;
            }
        }
        
        // If we have more data, use it to set end_dim within valid range
        if (offset + 1 <= Size) {
            int8_t raw_val = static_cast<int8_t>(Data[offset]);
            offset += 1;
            // Map to range [-ndim, ndim-1] for valid PyTorch indexing
            if (ndim > 0) {
                end_dim = raw_val % ndim;
            }
        }
        
        // Inner try-catch for expected failures (dimension ordering issues)
        try {
            // Create the Flatten module with options using builder pattern
            torch::nn::FlattenOptions options;
            options.start_dim(start_dim);
            options.end_dim(end_dim);
            torch::nn::Flatten flatten_module(options);
            
            // Apply the Flatten operation using the module
            torch::Tensor output = flatten_module->forward(input);
            
            // Ensure the output is valid by accessing some property
            auto output_sizes = output.sizes();
            (void)output_sizes;
        } catch (const c10::Error&) {
            // Expected for invalid dimension combinations, continue to try functional API
        }
        
        // Inner try-catch for functional API test
        try {
            // Alternative approach: use the functional API
            torch::Tensor output2 = torch::flatten(input, start_dim, end_dim);
            
            // Access output to ensure computation
            auto numel = output2.numel();
            (void)numel;
        } catch (const c10::Error&) {
            // Expected for invalid dimension combinations
        }
        
        // Also test with default parameters (common use case)
        try {
            torch::nn::Flatten flatten_default;
            torch::Tensor output_default = flatten_default->forward(input);
            (void)output_default.numel();
        } catch (const c10::Error&) {
            // May fail for 0-d or 1-d tensors
        }
        
        // Test with explicit common configurations
        try {
            // Flatten all dimensions
            torch::Tensor flat_all = torch::flatten(input);
            (void)flat_all.sizes();
        } catch (const c10::Error&) {
            // Expected in some cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}