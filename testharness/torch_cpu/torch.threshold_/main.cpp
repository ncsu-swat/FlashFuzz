#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For isnan, isinf

// --- Fuzzer Entry Point ---
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
        
        // Extract threshold and value parameters from the remaining data
        float threshold = 0.0f;
        float value = 0.0f;
        
        // Parse threshold value if we have enough data
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&threshold, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize to avoid NaN/Inf issues
            if (std::isnan(threshold) || std::isinf(threshold)) {
                threshold = 0.0f;
            }
        }
        
        // Parse replacement value if we have enough data
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize to avoid NaN/Inf issues
            if (std::isnan(value) || std::isinf(value)) {
                value = 0.0f;
            }
        }
        
        // Apply threshold_ operation (inplace)
        // threshold_ replaces values below threshold with the given value
        torch::threshold_(input, threshold, value);
        
        // Test with another tensor if we have more data
        if (offset + 1 <= Size) {
            uint8_t test_more = Data[offset++];
            
            if (test_more % 2 == 0) {
                // Create another tensor and test threshold_
                size_t new_offset = offset;
                torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, new_offset);
                
                // Use different threshold/value based on remaining data
                float threshold2 = (test_more / 2) * 0.1f - 5.0f;  // Range roughly -5 to 7.5
                float value2 = (test_more % 10) * 0.5f - 2.5f;     // Range roughly -2.5 to 2
                
                torch::threshold_(another_input, threshold2, value2);
            }
        }
        
        // Also test the non-inplace version for coverage
        if (offset + 1 <= Size) {
            uint8_t test_non_inplace = Data[offset++];
            
            if (test_non_inplace % 3 == 0) {
                size_t new_offset = offset;
                torch::Tensor test_input = fuzzer_utils::createTensor(Data, Size, new_offset);
                
                // Non-inplace threshold
                torch::Tensor output = torch::threshold(test_input, threshold, value);
                (void)output;  // Prevent unused variable warning
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}