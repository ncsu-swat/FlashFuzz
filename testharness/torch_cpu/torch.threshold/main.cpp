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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract threshold value from the input data
        float threshold_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&threshold_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Extract replacement value from the input data
        float replacement_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&replacement_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Handle NaN/Inf in threshold and replacement values
        if (std::isnan(threshold_value) || std::isinf(threshold_value)) {
            threshold_value = 0.0f;
        }
        if (std::isnan(replacement_value) || std::isinf(replacement_value)) {
            replacement_value = 0.0f;
        }
        
        // Apply threshold operation
        // y = x if x > threshold else value
        torch::Tensor output = torch::threshold(input, threshold_value, replacement_value);
        
        // Try in-place version using torch::threshold_
        try {
            torch::Tensor input_copy = input.clone();
            torch::threshold_(input_copy, threshold_value, replacement_value);
        } catch (...) {
            // Silently ignore - shape/type constraints may fail
        }
        
        // Try functional version with different threshold and replacement values
        if (offset + sizeof(float) <= Size) {
            float alt_threshold = 0.0f;
            std::memcpy(&alt_threshold, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            if (std::isnan(alt_threshold) || std::isinf(alt_threshold)) {
                alt_threshold = 0.5f;
            }
            
            float alt_replacement = 0.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&alt_replacement, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (std::isnan(alt_replacement) || std::isinf(alt_replacement)) {
                    alt_replacement = -1.0f;
                }
            }
            
            torch::Tensor output2 = torch::threshold(input, alt_threshold, alt_replacement);
        }
        
        // Try with different tensor types
        if (offset < Size) {
            torch::Tensor alt_input = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                torch::Tensor alt_output = torch::threshold(alt_input, threshold_value, replacement_value);
            } catch (...) {
                // Silently ignore potential type/shape issues
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_output = torch::threshold(empty_tensor, threshold_value, replacement_value);
        } catch (...) {
            // Silently ignore - empty tensor may not be supported
        }
        
        // Try with scalar tensor
        try {
            torch::Tensor scalar_tensor = torch::tensor(1.0f);
            torch::Tensor scalar_output = torch::threshold(scalar_tensor, threshold_value, replacement_value);
        } catch (...) {
            // Silently ignore
        }
        
        // Try with different dtypes
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::Tensor double_output = torch::threshold(double_input, static_cast<double>(threshold_value), static_cast<double>(replacement_value));
        } catch (...) {
            // Silently ignore
        }
        
        // Try with multi-dimensional tensor
        try {
            torch::Tensor reshaped = input.view({-1});
            torch::Tensor reshaped_output = torch::threshold(reshaped, threshold_value, replacement_value);
        } catch (...) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}