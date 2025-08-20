#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract threshold and value parameters from the remaining data
        float threshold = 0.0f;
        float value = 0.0f;
        
        // Parse threshold value if we have enough data
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&threshold, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Parse replacement value if we have enough data
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Create a copy of the input tensor for testing the inplace operation
        torch::Tensor input_copy = input.clone();
        
        // Apply threshold_ operation (inplace)
        torch::threshold_(input, threshold, value);
        
        // Also test the non-inplace version to ensure consistency
        if (offset + 1 <= Size) {
            // Use one more byte to decide whether to test the non-inplace version
            uint8_t test_non_inplace = Data[offset++];
            
            if (test_non_inplace % 2 == 0) {
                torch::Tensor output = torch::threshold(input_copy, threshold, value);
                
                // Optionally verify that inplace and non-inplace versions produce the same result
                // This is a sanity check, not a fuzzing target
                if (input.sizes() == output.sizes() && input.dtype() == output.dtype()) {
                    bool equal = torch::allclose(input, output);
                    if (!equal) {
                        throw std::runtime_error("Inplace and non-inplace threshold operations produced different results");
                    }
                }
            }
        }
        
        // Test threshold_ with different tensor types if we have more data
        if (offset + 1 <= Size) {
            uint8_t test_different_type = Data[offset++];
            
            if (test_different_type % 3 == 0) {
                // Create a tensor with a different data type
                size_t new_offset = offset;
                torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, new_offset);
                
                // Try to apply threshold_ to this tensor too
                torch::threshold_(another_input, threshold, value);
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