#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
            // If we don't have enough data for a second tensor, create a simple one
            divisor = torch::ones_like(input);
        }
        
        // Try different variants of fmod
        
        // 1. Tensor-Tensor fmod
        torch::Tensor result1 = torch::fmod(input, divisor);
        
        // 2. Tensor-Scalar fmod
        double scalar_value = 2.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        torch::Tensor result2 = torch::fmod(input, scalar_value);
        
        // 3. Create scalar tensor for scalar-tensor operations
        torch::Tensor scalar_tensor = torch::full_like(input, scalar_value);
        torch::Tensor result3 = torch::fmod(scalar_tensor, input);
        
        // 4. In-place fmod
        torch::Tensor input_copy = input.clone();
        input_copy.fmod_(divisor);
        
        // 5. In-place fmod with scalar
        torch::Tensor input_copy2 = input.clone();
        input_copy2.fmod_(scalar_value);
        
        // 6. Try with different dtypes if possible
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
            // These operations should work fine for floating point types
            torch::Tensor result4 = torch::fmod(input, 3.14);
            torch::Tensor scalar_tensor2 = torch::full_like(input, 2.71);
            torch::Tensor result5 = torch::fmod(scalar_tensor2, input);
        }
        
        // 7. Try with integer types
        if (input.dtype() == torch::kInt || input.dtype() == torch::kLong) {
            torch::Tensor result6 = torch::fmod(input, 7);
            torch::Tensor scalar_tensor3 = torch::full_like(input, 9);
            torch::Tensor result7 = torch::fmod(scalar_tensor3, input);
        }
        
        // 8. Try with zero divisor (should trigger division by zero)
        torch::Tensor zero_tensor = torch::zeros_like(divisor);
        try {
            torch::Tensor result_zero_div = torch::fmod(input, zero_tensor);
        } catch (const std::exception& e) {
            // Expected exception for division by zero
        }
        
        // 9. Try with broadcasting
        if (input.dim() > 0) {
            torch::Tensor small_tensor = torch::ones({1});
            torch::Tensor result_broadcast = torch::fmod(input, small_tensor);
        }
        
        // 10. Try with different shapes that should trigger broadcasting
        std::vector<int64_t> new_shape;
        if (input.dim() > 0) {
            new_shape.push_back(1);
            for (int i = 1; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            
            if (new_shape.size() > 0) {
                torch::Tensor broadcast_tensor = torch::ones(new_shape);
                torch::Tensor result_broadcast2 = torch::fmod(input, broadcast_tensor);
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