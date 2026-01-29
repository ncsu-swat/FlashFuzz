#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For isnan, isinf

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
        
        // Need at least a few bytes for the input tensor and threshold parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract threshold value and value to replace with
        double threshold = 0.0;
        double value = 0.0;
        
        // Extract threshold from the input data if available
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&threshold, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize the threshold value
            if (std::isnan(threshold) || std::isinf(threshold)) {
                threshold = 0.0;
            }
            // Clamp to reasonable range
            threshold = std::max(-1e6, std::min(1e6, threshold));
        }
        
        // Extract value from the input data if available
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize the value
            if (std::isnan(value) || std::isinf(value)) {
                value = 0.0;
            }
            // Clamp to reasonable range
            value = std::max(-1e6, std::min(1e6, value));
        }
        
        // Create Threshold module with ThresholdOptions
        torch::nn::ThresholdOptions options(threshold, value);
        torch::nn::Threshold threshold_module(options);
        
        // Apply threshold operation
        torch::Tensor output = threshold_module->forward(input);
        
        // Try inplace version as well if there's enough data left
        if (offset < Size) {
            bool inplace = Data[offset++] % 2 == 0;
            if (inplace) {
                try {
                    torch::Tensor input_copy = input.clone();
                    torch::threshold_(input_copy, threshold, value);
                } catch (...) {
                    // Silently ignore inplace failures
                }
            }
        }
        
        // Try with different threshold and value if there's enough data left
        if (offset + 2*sizeof(double) <= Size) {
            double new_threshold, new_value;
            std::memcpy(&new_threshold, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&new_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Sanitize new values
            if (std::isnan(new_threshold) || std::isinf(new_threshold)) {
                new_threshold = 0.5;
            }
            new_threshold = std::max(-1e6, std::min(1e6, new_threshold));
            
            if (std::isnan(new_value) || std::isinf(new_value)) {
                new_value = 0.0;
            }
            new_value = std::max(-1e6, std::min(1e6, new_value));
            
            // Create a new module with the new options
            torch::nn::ThresholdOptions new_options(new_threshold, new_value);
            torch::nn::Threshold threshold_module2(new_options);
            
            torch::Tensor output2 = threshold_module2->forward(input);
        }
        
        // Try functional version
        torch::Tensor functional_output = torch::threshold(input, threshold, value);
        
        // Try inplace functional version
        if (offset < Size) {
            bool inplace_functional = Data[offset++] % 2 == 0;
            if (inplace_functional) {
                try {
                    torch::Tensor input_copy = input.clone();
                    torch::threshold_(input_copy, threshold, value);
                } catch (...) {
                    // Silently ignore inplace failures
                }
            }
        }
        
        // Test with inplace option in the module
        if (offset < Size) {
            bool use_inplace_module = Data[offset++] % 2 == 0;
            if (use_inplace_module) {
                try {
                    torch::nn::ThresholdOptions inplace_options(threshold, value);
                    inplace_options.inplace(true);
                    torch::nn::Threshold inplace_module(inplace_options);
                    
                    torch::Tensor input_copy = input.clone();
                    torch::Tensor inplace_output = inplace_module->forward(input_copy);
                } catch (...) {
                    // Silently ignore inplace module failures
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}