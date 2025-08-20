#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get max value from remaining data
        float max_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Create a copy of the input tensor for testing the in-place operation
        torch::Tensor original = input_tensor.clone();
        
        // Apply clamp_max_ operation (in-place)
        input_tensor.clamp_max_(max_value);
        
        // Verify the operation worked correctly (all values should be <= max_value)
        torch::Tensor verification = original.clamp_max(max_value);
        
        // Optional: Check if the in-place operation matches the non-in-place version
        if (!torch::allclose(input_tensor, verification)) {
            throw std::runtime_error("In-place and out-of-place clamp_max operations produced different results");
        }
        
        // Test edge cases if we have more data
        if (offset + sizeof(float) <= Size) {
            // Try with NaN as max value
            float nan_value = std::numeric_limits<float>::quiet_NaN();
            torch::Tensor nan_test = original.clone();
            nan_test.clamp_max_(nan_value);
            
            // Try with infinity as max value
            float inf_value = std::numeric_limits<float>::infinity();
            torch::Tensor inf_test = original.clone();
            inf_test.clamp_max_(inf_value);
            
            // Try with negative infinity as max value
            float neg_inf_value = -std::numeric_limits<float>::infinity();
            torch::Tensor neg_inf_test = original.clone();
            neg_inf_test.clamp_max_(neg_inf_value);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}