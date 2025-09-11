#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::min

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create divisor tensor
        torch::Tensor divisor;
        if (offset < Size) {
            divisor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create a scalar tensor
            divisor = torch::tensor(1.0, input.options());
        }
        
        // Try different variants of remainder operation
        
        // 1. Tensor.remainder(other)
        torch::Tensor result1 = input.remainder(divisor);
        
        // 2. torch::remainder(input, other)
        torch::Tensor result2 = torch::remainder(input, divisor);
        
        // 3. torch::fmod(input, other) - similar operation to test
        torch::Tensor result3 = torch::fmod(input, divisor);
        
        // 4. Test with scalar divisor
        if (offset < Size) {
            // Use a byte from the input data as a scalar
            double scalar_value = static_cast<double>(Data[offset % Size]);
            torch::Tensor result4 = input.remainder(scalar_value);
            
            // 5. Test in-place version
            torch::Tensor input_copy = input.clone();
            input_copy.remainder_(scalar_value);
        }
        
        // 6. Test with different scalar types
        if (offset + 1 < Size) {
            uint8_t scalar_type_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(scalar_type_selector);
            
            // Convert input to the selected data type
            torch::Tensor typed_input = input.to(dtype);
            torch::Tensor typed_divisor = divisor.to(dtype);
            
            // Test remainder with the converted tensors
            torch::Tensor typed_result = torch::remainder(typed_input, typed_divisor);
        }
        
        // 7. Test with zero divisor (should trigger division by zero)
        torch::Tensor zero_divisor = torch::zeros_like(divisor);
        try {
            torch::Tensor zero_result = torch::remainder(input, zero_divisor);
        } catch (const std::exception& e) {
            // Expected exception for division by zero
        }
        
        // 8. Test with broadcasting
        if (input.dim() > 0 && divisor.dim() > 0) {
            // Create a smaller tensor for broadcasting
            std::vector<int64_t> smaller_shape;
            for (int i = 0; i < std::min(static_cast<int>(input.dim()), 2); i++) {
                smaller_shape.push_back(1);
            }
            
            torch::Tensor broadcast_divisor = torch::ones(smaller_shape, divisor.options());
            torch::Tensor broadcast_result = torch::remainder(input, broadcast_divisor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
