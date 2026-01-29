#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
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
        // Need at least a few bytes for basic operations
        if (Size < 12) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract threshold value and replacement value from the input data first
        float threshold_value = 0.0f;
        float value = 0.0f;
        
        std::memcpy(&threshold_value, Data + offset, sizeof(float));
        offset += sizeof(float);
        
        std::memcpy(&value, Data + offset, sizeof(float));
        offset += sizeof(float);
        
        // Sanitize float values to avoid NaN/Inf propagation issues
        if (!std::isfinite(threshold_value)) {
            threshold_value = 0.0f;
        }
        if (!std::isfinite(value)) {
            value = 0.0f;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Threshold module
        torch::nn::Threshold threshold_module(threshold_value, value);
        
        // Apply the threshold operation
        torch::Tensor output = threshold_module->forward(input);
        
        // Try inplace version if there's more data
        if (offset < Size && Data[offset] % 2 == 0) {
            offset++;
            try {
                torch::Tensor input_clone = input.clone();
                torch::nn::ThresholdOptions options(threshold_value, value);
                options.inplace(true);
                torch::nn::Threshold inplace_threshold_module(options);
                inplace_threshold_module->forward(input_clone);
            } catch (...) {
                // Silently ignore inplace failures
            }
        } else if (offset < Size) {
            offset++;
        }
        
        // Try with different threshold and value parameters
        if (offset < Size) {
            uint8_t create_new = Data[offset++];
            
            if (create_new % 3 == 0) {
                // Create a new threshold module with different parameters
                float new_threshold = -threshold_value;
                float new_value = value * 2.0f;
                
                if (!std::isfinite(new_value)) {
                    new_value = value;
                }
                
                torch::nn::Threshold new_threshold_module(new_threshold, new_value);
                torch::Tensor new_output = new_threshold_module->forward(input);
            }
        }
        
        // Try with edge case values
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            try {
                if (edge_case % 5 == 0) {
                    // Try with large positive threshold
                    torch::nn::Threshold large_threshold(1e6f, 0.0f);
                    torch::Tensor large_output = large_threshold->forward(input);
                } else if (edge_case % 5 == 1) {
                    // Try with large negative threshold
                    torch::nn::Threshold small_threshold(-1e6f, 0.0f);
                    torch::Tensor small_output = small_threshold->forward(input);
                } else if (edge_case % 5 == 2) {
                    // Try with zero threshold
                    torch::nn::Threshold zero_threshold(0.0f, -1.0f);
                    torch::Tensor zero_output = zero_threshold->forward(input);
                } else if (edge_case % 5 == 3) {
                    // Try with threshold equal to value
                    torch::nn::Threshold eq_threshold(1.0f, 1.0f);
                    torch::Tensor eq_output = eq_threshold->forward(input);
                } else {
                    // Try with negative value replacement
                    torch::nn::Threshold neg_val_threshold(0.5f, -999.0f);
                    torch::Tensor neg_output = neg_val_threshold->forward(input);
                }
            } catch (...) {
                // Silently ignore edge case failures
            }
        }
        
        // Test with different tensor types if we have remaining data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            try {
                torch::Tensor typed_input;
                switch (dtype_selector % 4) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input.to(torch::kFloat16);
                        break;
                    default:
                        typed_input = input;
                        break;
                }
                torch::Tensor typed_output = threshold_module->forward(typed_input);
            } catch (...) {
                // Some dtypes may not be supported
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