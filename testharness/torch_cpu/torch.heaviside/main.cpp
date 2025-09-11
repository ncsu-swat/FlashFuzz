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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create values tensor
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create a simple one
            values = torch::ones_like(input);
        }
        
        // Apply heaviside operation
        torch::Tensor result = torch::heaviside(input, values);
        
        // Try different variants of the operation
        if (Size % 3 == 0) {
            // Test with scalar values
            torch::Tensor scalar_values = torch::full_like(input, 0.5);
            torch::Tensor result2 = torch::heaviside(input, scalar_values);
        } else if (Size % 3 == 1) {
            // Test with out parameter
            torch::Tensor out = torch::empty_like(input);
            torch::heaviside_out(out, input, values);
        } else {
            // Test with different shapes that can broadcast
            if (input.dim() > 0) {
                auto shape = input.sizes().vec();
                if (shape[0] > 1) {
                    shape[0] = 1;  // Make first dimension 1 for broadcasting
                    torch::Tensor broadcast_values = torch::ones(shape, values.options());
                    torch::Tensor result3 = torch::heaviside(input, broadcast_values);
                }
            }
        }
        
        // Test edge cases with special values
        if (Size % 5 == 0) {
            // Test with NaN values
            torch::Tensor nan_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
            torch::Tensor nan_result = torch::heaviside(nan_input, values);
        } else if (Size % 5 == 1) {
            // Test with infinity values
            torch::Tensor inf_input = torch::full_like(input, std::numeric_limits<float>::infinity());
            torch::Tensor inf_result = torch::heaviside(inf_input, values);
        } else if (Size % 5 == 2) {
            // Test with negative infinity values
            torch::Tensor neg_inf_input = torch::full_like(input, -std::numeric_limits<float>::infinity());
            torch::Tensor neg_inf_result = torch::heaviside(neg_inf_input, values);
        } else if (Size % 5 == 3) {
            // Test with zero values
            torch::Tensor zero_input = torch::zeros_like(input);
            torch::Tensor zero_result = torch::heaviside(zero_input, values);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
