#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract threshold value and replacement value from the input data
        float threshold_value = 0.0f;
        float value = 0.0f;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&threshold_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Create Threshold module
        torch::nn::Threshold threshold_module(threshold_value, value);
        
        // Apply the threshold operation
        torch::Tensor output = threshold_module->forward(input);
        
        // Try inplace version if there's more data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_clone = input.clone();
            torch::nn::ThresholdOptions options(threshold_value, value);
            options.inplace(true);
            torch::nn::Threshold inplace_threshold_module(options);
            inplace_threshold_module->forward(input_clone);
        }
        
        // Try with different threshold and value parameters
        if (offset + 1 < Size) {
            // Use the next byte to determine if we should create a new module
            uint8_t create_new = Data[offset++];
            
            if (create_new % 3 == 0) {
                // Create a new threshold module with different parameters
                float new_threshold = threshold_value * -1.0f; // Try negative threshold
                float new_value = value * 2.0f;                // Try different value
                
                torch::nn::Threshold new_threshold_module(new_threshold, new_value);
                torch::Tensor new_output = new_threshold_module->forward(input);
            }
        }
        
        // Try with edge case values
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            if (edge_case % 5 == 0) {
                // Try with very large threshold
                torch::nn::Threshold large_threshold(1e10, 0.0f);
                torch::Tensor large_output = large_threshold->forward(input);
            } else if (edge_case % 5 == 1) {
                // Try with very small threshold
                torch::nn::Threshold small_threshold(-1e10, 0.0f);
                torch::Tensor small_output = small_threshold->forward(input);
            } else if (edge_case % 5 == 2) {
                // Try with NaN threshold
                torch::nn::Threshold nan_threshold(std::numeric_limits<float>::quiet_NaN(), 0.0f);
                torch::Tensor nan_output = nan_threshold->forward(input);
            } else if (edge_case % 5 == 3) {
                // Try with infinity threshold
                torch::nn::Threshold inf_threshold(std::numeric_limits<float>::infinity(), 0.0f);
                torch::Tensor inf_output = inf_threshold->forward(input);
            } else {
                // Try with same threshold and value
                torch::nn::Threshold same_threshold(1.0f, 1.0f);
                torch::Tensor same_output = same_threshold->forward(input);
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
