#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Get dimensions from fuzzer data for Linear module
        int64_t in_features = static_cast<int64_t>((Data[offset++] % 15) + 1);  // 1-16
        int64_t out_features = static_cast<int64_t>((Data[offset++] % 15) + 1); // 1-16
        
        // Create a simple linear module
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features));
        
        // Apply weight norm to the module
        torch::nn::utils::weight_norm(linear, torch::nn::utils::WeightNormOptions().name("weight"));
        
        // Try to remove weight norm
        torch::nn::utils::remove_weight_norm(linear, "weight");
        
        // Try with different dimension values
        if (offset + 1 < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset++] % 2);  // Valid dims for 2D weight: 0 or 1
            
            // Recreate linear module for fresh test
            torch::nn::Linear linear2(torch::nn::LinearOptions(in_features, out_features));
            
            // Apply weight norm with dimension
            try {
                torch::nn::utils::weight_norm(linear2, torch::nn::utils::WeightNormOptions().name("weight").dim(dim));
                
                // Remove weight norm
                torch::nn::utils::remove_weight_norm(linear2, "weight");
            } catch (...) {
                // May fail for certain dim values, continue
            }
        }
        
        // Try with a non-existent name (expected to fail)
        if (offset < Size) {
            torch::nn::Linear linear3(torch::nn::LinearOptions(in_features, out_features));
            try {
                torch::nn::utils::remove_weight_norm(linear3, "non_existent_weight");
            } catch (...) {
                // Expected to fail, continue
            }
        }
        
        // Try removing weight norm when it's not applied (expected to fail)
        {
            torch::nn::Linear linear4(torch::nn::LinearOptions(in_features, out_features));
            try {
                torch::nn::utils::remove_weight_norm(linear4, "weight");
            } catch (...) {
                // Expected to fail since weight norm was never applied
            }
        }
        
        // Create a Conv2d module and test with it
        if (offset + 2 < Size) {
            int64_t in_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);   // 1-4
            int64_t out_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);  // 1-4
            int64_t kernel_size = static_cast<int64_t>((Data[offset++] % 3) + 1);   // 1-3
            
            torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
            
            // Apply weight norm
            torch::nn::utils::weight_norm(conv, torch::nn::utils::WeightNormOptions().name("weight"));
            
            // Remove weight norm
            torch::nn::utils::remove_weight_norm(conv, "weight");
            
            // Test with bias if available
            if (conv->options.bias()) {
                torch::nn::Conv2d conv2(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
                try {
                    torch::nn::utils::weight_norm(conv2, torch::nn::utils::WeightNormOptions().name("bias"));
                    torch::nn::utils::remove_weight_norm(conv2, "bias");
                } catch (...) {
                    // May fail depending on implementation
                }
            }
        }
        
        // Test with Conv1d
        if (offset + 2 < Size) {
            int64_t in_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);
            int64_t out_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);
            int64_t kernel_size = static_cast<int64_t>((Data[offset++] % 3) + 1);
            
            torch::nn::Conv1d conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size));
            
            try {
                torch::nn::utils::weight_norm(conv1d, torch::nn::utils::WeightNormOptions().name("weight"));
                torch::nn::utils::remove_weight_norm(conv1d, "weight");
            } catch (...) {
                // Continue on failure
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}