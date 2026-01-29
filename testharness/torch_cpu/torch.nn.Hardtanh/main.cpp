#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Hardtanh from the remaining data
        float min_val = -1.0f;
        float max_val = 1.0f;
        
        // If we have more data, use it for min_val and max_val
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize to avoid NaN/Inf issues
            if (!std::isfinite(min_val)) {
                min_val = -1.0f;
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize to avoid NaN/Inf issues
            if (!std::isfinite(max_val)) {
                max_val = 1.0f;
            }
        }
        
        // Ensure min_val <= max_val for valid operation
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // Create Hardtanh module
        torch::nn::Hardtanh hardtanh(torch::nn::HardtanhOptions().min_val(min_val).max_val(max_val));
        
        // Apply Hardtanh to the input tensor
        torch::Tensor output = hardtanh->forward(input);
        
        // Try functional version as well
        try {
            torch::Tensor output_functional = torch::nn::functional::hardtanh(
                input, 
                torch::nn::functional::HardtanhFuncOptions().min_val(min_val).max_val(max_val)
            );
        } catch (...) {
            // Silently ignore expected failures in functional version
        }
        
        // Try inplace version using hardtanh with inplace option
        try {
            torch::Tensor input_copy = input.clone();
            torch::nn::functional::hardtanh(
                input_copy,
                torch::nn::functional::HardtanhFuncOptions().min_val(min_val).max_val(max_val).inplace(true)
            );
        } catch (...) {
            // Silently ignore expected failures in inplace version
        }
        
        // Try with default parameters
        torch::nn::Hardtanh default_hardtanh;
        torch::Tensor output_default = default_hardtanh->forward(input);
        
        // Try with edge case parameters
        if (offset + 2*sizeof(float) <= Size) {
            float edge_min, edge_max;
            std::memcpy(&edge_min, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&edge_max, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Sanitize edge values
            if (!std::isfinite(edge_min)) edge_min = -1.0f;
            if (!std::isfinite(edge_max)) edge_max = 1.0f;
            
            try {
                // Try with potentially problematic min/max values
                torch::nn::Hardtanh edge_hardtanh(torch::nn::HardtanhOptions().min_val(edge_min).max_val(edge_max));
                torch::Tensor output_edge = edge_hardtanh->forward(input);
            } catch (...) {
                // Silently ignore edge case failures
            }
            
            try {
                // Try with swapped values
                torch::nn::Hardtanh swapped_hardtanh(torch::nn::HardtanhOptions().min_val(edge_max).max_val(edge_min));
                torch::Tensor output_swapped = swapped_hardtanh->forward(input);
            } catch (...) {
                // Silently ignore swapped case failures
            }
        }
        
        // Test with different tensor dtypes
        try {
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::Tensor output_float = hardtanh->forward(float_input);
        } catch (...) {
            // Silently ignore dtype conversion failures
        }
        
        try {
            torch::Tensor double_input = input.to(torch::kFloat64);
            torch::Tensor output_double = hardtanh->forward(double_input);
        } catch (...) {
            // Silently ignore dtype conversion failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}