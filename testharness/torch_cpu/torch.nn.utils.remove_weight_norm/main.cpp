#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use as weights
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple linear module
        torch::nn::Linear linear(weight.size(0), weight.size(0));
        
        // Apply weight norm to the module
        torch::nn::utils::weight_norm(linear, torch::nn::utils::WeightNormOptions().name("weight"));
        
        // Try to remove weight norm
        torch::nn::utils::remove_weight_norm(linear, "weight");
        
        // Try with different dimension values
        if (offset + 1 < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset++]);
            
            // Apply weight norm with dimension
            torch::nn::utils::weight_norm(linear, torch::nn::utils::WeightNormOptions().name("weight").dim(dim));
            
            // Remove weight norm
            torch::nn::utils::remove_weight_norm(linear, "weight");
        }
        
        // Try with a non-existent name
        if (offset < Size) {
            try {
                torch::nn::utils::remove_weight_norm(linear, "non_existent_weight");
            } catch (...) {
                // Expected to fail, continue
            }
        }
        
        // Try removing weight norm when it's not applied
        try {
            torch::nn::utils::remove_weight_norm(linear, "weight");
        } catch (...) {
            // May fail if weight norm was already removed
        }
        
        // Create a Conv2d module and test with it
        if (offset < Size) {
            int64_t in_channels = 3;
            int64_t out_channels = 3;
            int64_t kernel_size = 3;
            
            torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
            
            // Apply weight norm
            torch::nn::utils::weight_norm(conv, torch::nn::utils::WeightNormOptions().name("weight"));
            
            // Remove weight norm
            torch::nn::utils::remove_weight_norm(conv, "weight");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}