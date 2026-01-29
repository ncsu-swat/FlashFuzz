#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <limits>

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
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty tensors for certain operations
        if (input.numel() == 0) {
            return 0;
        }
        
        // Extract min_val and max_val from the remaining data
        float min_val = -1.0f;
        float max_val = 1.0f;
        
        if (offset + sizeof(float) <= Size) {
            float extracted_val;
            std::memcpy(&extracted_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize: avoid NaN/Inf in parameters
            if (std::isfinite(extracted_val)) {
                min_val = extracted_val;
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float extracted_val;
            std::memcpy(&extracted_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize: avoid NaN/Inf in parameters
            if (std::isfinite(extracted_val)) {
                max_val = extracted_val;
            }
        }
        
        // Ensure min_val <= max_val for valid Hardtanh
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // Create Hardtanh module with fuzzed parameters
        torch::nn::Hardtanh hardtanh_module(
            torch::nn::HardtanhOptions().min_val(min_val).max_val(max_val)
        );
        
        // Apply Hardtanh to the input tensor
        torch::Tensor output = hardtanh_module->forward(input);
        
        // Try functional version as well
        torch::Tensor output_functional = torch::nn::functional::hardtanh(
            input, 
            torch::nn::functional::HardtanhFuncOptions().min_val(min_val).max_val(max_val)
        );
        
        // Try inplace version
        if (offset < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag % 2 == 0) {
                torch::Tensor input_clone = input.clone();
                torch::nn::functional::hardtanh(
                    input_clone, 
                    torch::nn::functional::HardtanhFuncOptions()
                        .min_val(min_val)
                        .max_val(max_val)
                        .inplace(true)
                );
            }
        }
        
        // Try with default parameters
        torch::nn::Hardtanh default_hardtanh;
        torch::Tensor output_default = default_hardtanh->forward(input);
        
        // Try with edge case parameters
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            try {
                if (edge_case % 4 == 0) {
                    // Case with very large values
                    torch::nn::Hardtanh large_hardtanh(
                        torch::nn::HardtanhOptions().min_val(-1e10).max_val(1e10)
                    );
                    torch::Tensor output_large = large_hardtanh->forward(input);
                } else if (edge_case % 4 == 1) {
                    // Case with very small range
                    torch::nn::Hardtanh small_hardtanh(
                        torch::nn::HardtanhOptions().min_val(-1e-10).max_val(1e-10)
                    );
                    torch::Tensor output_small = small_hardtanh->forward(input);
                } else if (edge_case % 4 == 2) {
                    // Case with equal min and max (clamps all values to single point)
                    torch::nn::Hardtanh equal_hardtanh(
                        torch::nn::HardtanhOptions().min_val(0.0).max_val(0.0)
                    );
                    torch::Tensor output_equal = equal_hardtanh->forward(input);
                } else {
                    // Test with different dtype inputs
                    if (input.scalar_type() == torch::kFloat) {
                        torch::Tensor double_input = input.to(torch::kDouble);
                        torch::Tensor output_double = hardtanh_module->forward(double_input);
                    }
                }
            } catch (const std::exception &) {
                // Silently catch expected failures from edge cases
            }
        }
        
        // Test Hardtanh6 variant (commonly used ReLU6-like activation)
        try {
            torch::nn::Hardtanh hardtanh6(
                torch::nn::HardtanhOptions().min_val(0.0).max_val(6.0)
            );
            torch::Tensor output6 = hardtanh6->forward(input);
        } catch (const std::exception &) {
            // Silently catch
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}