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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for adaptive modules
        int64_t output_size = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            output_size = std::abs(output_size) % 16 + 1; // Ensure positive and reasonable size
        } else {
            output_size = 3; // Default value
        }
        
        // Test AdaptiveAvgPool1d
        if (input.dim() >= 2) {
            try {
                auto m1 = torch::nn::AdaptiveAvgPool1d(output_size);
                auto result1 = m1->forward(input);
            } catch (...) {
                // Continue with other tests
            }
        }
        
        // Test AdaptiveAvgPool2d
        if (input.dim() >= 3) {
            try {
                // Test with single output size
                auto m2a = torch::nn::AdaptiveAvgPool2d(output_size);
                auto result2a = m2a->forward(input);
                
                // Test with tuple output size
                std::vector<int64_t> output_sizes = {output_size, output_size};
                auto m2b = torch::nn::AdaptiveAvgPool2d(output_sizes);
                auto result2b = m2b->forward(input);
            } catch (...) {
                // Continue with other tests
            }
        }
        
        // Test AdaptiveAvgPool3d
        if (input.dim() >= 4) {
            try {
                // Test with single output size
                auto m3a = torch::nn::AdaptiveAvgPool3d(output_size);
                auto result3a = m3a->forward(input);
                
                // Test with tuple output size
                std::vector<int64_t> output_sizes = {output_size, output_size, output_size};
                auto m3b = torch::nn::AdaptiveAvgPool3d(output_sizes);
                auto result3b = m3b->forward(input);
            } catch (...) {
                // Continue with other tests
            }
        }
        
        // Test AdaptiveMaxPool1d
        if (input.dim() >= 2) {
            try {
                auto m4 = torch::nn::AdaptiveMaxPool1d(output_size);
                auto result4 = m4->forward(input);
            } catch (...) {
                // Continue with other tests
            }
        }
        
        // Test AdaptiveMaxPool2d
        if (input.dim() >= 3) {
            try {
                // Test with single output size
                auto m5a = torch::nn::AdaptiveMaxPool2d(output_size);
                auto result5a = m5a->forward(input);
                
                // Test with tuple output size
                std::vector<int64_t> output_sizes = {output_size, output_size};
                auto m5b = torch::nn::AdaptiveMaxPool2d(output_sizes);
                auto result5b = m5b->forward(input);
            } catch (...) {
                // Continue with other tests
            }
        }
        
        // Test AdaptiveMaxPool3d
        if (input.dim() >= 4) {
            try {
                // Test with single output size
                auto m6a = torch::nn::AdaptiveMaxPool3d(output_size);
                auto result6a = m6a->forward(input);
                
                // Test with tuple output size
                std::vector<int64_t> output_sizes = {output_size, output_size, output_size};
                auto m6b = torch::nn::AdaptiveMaxPool3d(output_sizes);
                auto result6b = m6b->forward(input);
            } catch (...) {
                // Continue with other tests
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