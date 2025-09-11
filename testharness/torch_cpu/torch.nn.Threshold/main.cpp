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
        }
        
        // Extract value from the input data if available
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create Threshold module
        torch::nn::Threshold threshold_module(threshold, value);
        
        // Apply threshold operation
        torch::Tensor output = threshold_module->forward(input);
        
        // Try inplace version as well if there's enough data left
        if (offset < Size) {
            bool inplace = Data[offset++] % 2 == 0;
            if (inplace) {
                torch::Tensor input_copy = input.clone();
                torch::threshold_(input_copy, threshold, value);
            }
        }
        
        // Try with different threshold and value if there's enough data left
        if (offset + 2*sizeof(double) <= Size) {
            double new_threshold, new_value;
            std::memcpy(&new_threshold, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&new_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            threshold_module->options.threshold(new_threshold);
            threshold_module->options.value(new_value);
            
            torch::Tensor output2 = threshold_module->forward(input);
        }
        
        // Try functional version
        torch::Tensor functional_output = torch::threshold(input, threshold, value);
        
        // Try inplace functional version
        if (offset < Size) {
            bool inplace_functional = Data[offset++] % 2 == 0;
            if (inplace_functional) {
                torch::Tensor input_copy = input.clone();
                torch::threshold_(input_copy, threshold, value);
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
