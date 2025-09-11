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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get min value from remaining data
        float min_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&min_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Create a copy of the input tensor for testing the in-place operation
        torch::Tensor tensor_copy = input_tensor.clone();
        
        // Apply clamp_min_ operation (in-place)
        tensor_copy.clamp_min_(min_value);
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::clamp_min(input_tensor, min_value);
        
        // Optional: Verify results match between in-place and out-of-place versions
        if (!torch::allclose(tensor_copy, expected, 1e-5, 1e-8)) {
            throw std::runtime_error("In-place and out-of-place clamp_min results don't match");
        }
        
        // Test edge cases with different min values
        if (offset + sizeof(float) <= Size) {
            // Try with extreme min value
            float extreme_min;
            std::memcpy(&extreme_min, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Use very large or very small values
            if (std::abs(extreme_min) < 1e-6) {
                extreme_min = std::numeric_limits<float>::infinity();
            } else if (std::abs(extreme_min) > 1e6) {
                extreme_min = -std::numeric_limits<float>::infinity();
            }
            
            // Apply clamp_min_ with extreme value
            torch::Tensor extreme_copy = input_tensor.clone();
            extreme_copy.clamp_min_(extreme_min);
        }
        
        // Test with NaN as min value if we have more data
        if (offset + 1 < Size) {
            if (Data[offset] % 2 == 0) {
                torch::Tensor nan_copy = input_tensor.clone();
                nan_copy.clamp_min_(std::numeric_limits<float>::quiet_NaN());
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
