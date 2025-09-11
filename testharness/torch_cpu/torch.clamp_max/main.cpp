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
        
        // Need at least a few bytes for input tensor and max value
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract max value from remaining data
        double max_value = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        } else if (offset < Size) {
            // Use remaining bytes to create a scalar value
            uint8_t byte_value = Data[offset++];
            max_value = static_cast<double>(byte_value);
        }
        
        // Apply clamp_max in different ways
        
        // 1. Using torch::clamp_max directly
        torch::Tensor result1 = torch::clamp_max(input, max_value);
        
        // 2. Using the tensor's clamp_max method
        torch::Tensor result2 = input.clamp_max(max_value);
        
        // 3. Using out variant if there's enough data
        torch::Tensor out = torch::empty_like(input);
        torch::clamp_max_out(out, input, max_value);
        
        // 4. Using in-place variant
        torch::Tensor input_copy = input.clone();
        input_copy.clamp_max_(max_value);
        
        // 5. Try with a tensor max value if there's enough data left
        if (offset < Size) {
            torch::Tensor max_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Handle scalar tensor case
            if (max_tensor.dim() == 0) {
                torch::Tensor result_tensor_max = torch::clamp_max(input, max_tensor);
            } else if (max_tensor.sizes() == input.sizes()) {
                // Element-wise max values (broadcasting should work if shapes are compatible)
                torch::Tensor result_tensor_max = torch::clamp_max(input, max_tensor);
            }
        }
        
        // 6. Try with different scalar types for max value
        torch::Tensor result_int_max = torch::clamp_max(input, static_cast<int64_t>(max_value));
        torch::Tensor result_float_max = torch::clamp_max(input, static_cast<float>(max_value));
        
        // 7. Edge case: Try with extreme values
        torch::Tensor result_inf_max = torch::clamp_max(input, std::numeric_limits<double>::infinity());
        torch::Tensor result_neg_inf_max = torch::clamp_max(input, -std::numeric_limits<double>::infinity());
        
        // 8. Edge case: Try with NaN max value
        torch::Tensor result_nan_max = torch::clamp_max(input, std::numeric_limits<double>::quiet_NaN());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
