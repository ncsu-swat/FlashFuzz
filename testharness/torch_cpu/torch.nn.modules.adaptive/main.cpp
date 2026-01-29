#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Extract a second output size for asymmetric pooling
        int64_t output_size2 = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_size2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            output_size2 = std::abs(output_size2) % 16 + 1;
        } else {
            output_size2 = output_size;
        }
        
        // Test AdaptiveAvgPool1d (requires 2D or 3D input)
        if (input.dim() == 2 || input.dim() == 3) {
            try {
                auto m1 = torch::nn::AdaptiveAvgPool1d(output_size);
                auto result1 = m1->forward(input);
            } catch (...) {
                // Shape mismatch, continue
            }
        }
        
        // Test AdaptiveAvgPool2d (requires 3D or 4D input)
        if (input.dim() == 3 || input.dim() == 4) {
            try {
                // Test with single output size (square)
                auto m2a = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(output_size));
                auto result2a = m2a->forward(input);
            } catch (...) {
                // Continue
            }
            
            try {
                // Test with tuple output size (asymmetric)
                auto m2b = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({output_size, output_size2}));
                auto result2b = m2b->forward(input);
            } catch (...) {
                // Continue
            }
        }
        
        // Test AdaptiveAvgPool3d (requires 4D or 5D input)
        if (input.dim() == 4 || input.dim() == 5) {
            try {
                auto m3a = torch::nn::AdaptiveAvgPool3d(torch::nn::AdaptiveAvgPool3dOptions(output_size));
                auto result3a = m3a->forward(input);
            } catch (...) {
                // Continue
            }
            
            try {
                auto m3b = torch::nn::AdaptiveAvgPool3d(torch::nn::AdaptiveAvgPool3dOptions({output_size, output_size2, output_size}));
                auto result3b = m3b->forward(input);
            } catch (...) {
                // Continue
            }
        }
        
        // Test AdaptiveMaxPool1d (requires 2D or 3D input)
        if (input.dim() == 2 || input.dim() == 3) {
            try {
                auto m4 = torch::nn::AdaptiveMaxPool1d(output_size);
                auto result4 = m4->forward(input);
            } catch (...) {
                // Continue
            }
        }
        
        // Test AdaptiveMaxPool2d (requires 3D or 4D input)
        if (input.dim() == 3 || input.dim() == 4) {
            try {
                auto m5a = torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions(output_size));
                auto result5a = m5a->forward(input);
            } catch (...) {
                // Continue
            }
            
            try {
                auto m5b = torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({output_size, output_size2}));
                auto result5b = m5b->forward(input);
            } catch (...) {
                // Continue
            }
        }
        
        // Test AdaptiveMaxPool3d (requires 4D or 5D input)
        if (input.dim() == 4 || input.dim() == 5) {
            try {
                auto m6a = torch::nn::AdaptiveMaxPool3d(torch::nn::AdaptiveMaxPool3dOptions(output_size));
                auto result6a = m6a->forward(input);
            } catch (...) {
                // Continue
            }
            
            try {
                auto m6b = torch::nn::AdaptiveMaxPool3d(torch::nn::AdaptiveMaxPool3dOptions({output_size, output_size2, output_size}));
                auto result6b = m6b->forward(input);
            } catch (...) {
                // Continue
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