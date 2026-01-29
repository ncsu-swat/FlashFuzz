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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the tensor has at least 4 or 5 dimensions 
        // AdaptiveAvgPool3d expects (N, C, D, H, W) or (C, D, H, W)
        if (input.dim() < 4) {
            std::vector<int64_t> new_shape;
            
            // Keep original dimensions
            for (int i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            
            // Add missing dimensions to reach at least 4
            while (new_shape.size() < 4) {
                new_shape.push_back(1);
            }
            
            input = input.reshape(new_shape);
        }
        
        // Parse output size parameters from the remaining data
        std::vector<int64_t> output_size;
        
        // Try to get 3 values for D, H, W output sizes
        for (int i = 0; i < 3; i++) {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t size_val;
                std::memcpy(&size_val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Output size must be positive (at least 1)
                if (size_val <= 0) {
                    size_val = 1;
                } else if (size_val > 100) {
                    size_val = (size_val % 100) + 1;  // Limit to reasonable size
                }
                
                output_size.push_back(size_val);
            } else {
                // Default to 1 if not enough data
                output_size.push_back(1);
            }
        }
        
        // Create AdaptiveAvgPool3d module with different output size configurations
        torch::nn::AdaptiveAvgPool3d pool = nullptr;
        
        // Try different output size configurations
        uint8_t config_type = 0;
        if (offset < Size) {
            config_type = Data[offset++] % 4;
        }
        
        switch (config_type) {
            case 0:
                // Single integer for all dimensions
                pool = torch::nn::AdaptiveAvgPool3d(
                    torch::nn::AdaptiveAvgPool3dOptions(output_size[0]));
                break;
                
            case 1:
                // Tuple of 3 values
                pool = torch::nn::AdaptiveAvgPool3d(
                    torch::nn::AdaptiveAvgPool3dOptions(
                        std::vector<int64_t>{output_size[0], output_size[1], output_size[2]}));
                break;
                
            case 2:
                // Different values for each dimension
                pool = torch::nn::AdaptiveAvgPool3d(
                    torch::nn::AdaptiveAvgPool3dOptions(
                        std::vector<int64_t>{output_size[0], output_size[1], output_size[2]}));
                break;
                
            case 3:
            default:
                // Default: output size of 1
                pool = torch::nn::AdaptiveAvgPool3d(
                    torch::nn::AdaptiveAvgPool3dOptions(1));
                break;
        }
        
        // Apply the pooling operation
        // Inner try-catch for expected failures (shape mismatches, etc.)
        try {
            torch::Tensor output = pool->forward(input);
            
            // Basic validation that output has expected dimensions
            (void)output.sizes();
        } catch (const c10::Error&) {
            // Expected failures for invalid input shapes
        }
        
        // Test with 5D input (batch mode)
        if (input.dim() == 4 && offset < Size && Data[offset] % 2 == 0) {
            try {
                torch::Tensor input_5d = input.unsqueeze(0);  // Add batch dimension
                torch::Tensor output_5d = pool->forward(input_5d);
                (void)output_5d.sizes();
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }
        
        // Test with different dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_choice = Data[offset++] % 3;
            try {
                torch::Tensor typed_input;
                switch (dtype_choice) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                    default:
                        typed_input = input.to(torch::kFloat16);
                        break;
                }
                torch::Tensor typed_output = pool->forward(typed_input);
                (void)typed_output.sizes();
            } catch (const c10::Error&) {
                // Some dtypes may not be supported
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // keep the input
}