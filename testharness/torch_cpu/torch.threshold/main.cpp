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
        
        // Apply threshold operation
        // y = x if x > threshold else value
        torch::Tensor output = torch::threshold(input, threshold_value, replacement_value);
        
        // Try in-place version as well
        torch::Tensor input_copy = input.clone();
        torch::threshold_(input_copy, threshold_value, replacement_value);
        
        // Try functional version with different threshold and replacement values
        if (offset + sizeof(float) <= Size) {
            float alt_threshold = 0.0f;
            std::memcpy(&alt_threshold, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            float alt_replacement = 0.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&alt_replacement, Data + offset, sizeof(float));
                offset += sizeof(float);
            }
            
            torch::Tensor output2 = torch::threshold(input, alt_threshold, alt_replacement);
        }
        
        // Try with different tensor types
        if (offset < Size) {
            torch::Tensor alt_input = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor alt_output = torch::threshold(alt_input, threshold_value, replacement_value);
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        torch::Tensor empty_output = torch::threshold(empty_tensor, threshold_value, replacement_value);
        
        // Try with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(1.0f);
        torch::Tensor scalar_output = torch::threshold(scalar_tensor, threshold_value, replacement_value);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
