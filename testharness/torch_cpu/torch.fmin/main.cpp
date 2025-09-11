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
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, create a tensor with same shape but different values
            input2 = torch::ones_like(input1);
        }
        
        // Apply torch.fmin operation
        // fmin returns the minimum of each element in input1 and input2
        torch::Tensor result = torch::fmin(input1, input2);
        
        // Try broadcasting version if shapes are different
        if (input1.sizes() != input2.sizes()) {
            try {
                torch::Tensor broadcast_result = torch::fmin(input1, input2);
            } catch (const std::exception&) {
                // Broadcasting might fail for incompatible shapes, which is expected
            }
        }
        
        // Try scalar version
        if (offset + 1 < Size) {
            double scalar_value = static_cast<double>(Data[offset]);
            try {
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor scalar_result1 = torch::fmin(input1, scalar_tensor);
                torch::Tensor scalar_result2 = torch::fmin(scalar_tensor, input1);
            } catch (const std::exception&) {
                // Some scalar operations might fail depending on tensor type
            }
        }
        
        // Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_result = torch::fmin(empty_tensor, empty_tensor);
        } catch (const std::exception&) {
            // Empty tensor operations might fail
        }
        
        // Try with NaN values
        try {
            torch::Tensor nan_tensor = torch::full_like(input1, std::numeric_limits<float>::quiet_NaN());
            torch::Tensor nan_result = torch::fmin(input1, nan_tensor);
        } catch (const std::exception&) {
            // NaN operations might behave differently
        }
        
        // Try with infinity values
        try {
            torch::Tensor inf_tensor = torch::full_like(input1, std::numeric_limits<float>::infinity());
            torch::Tensor inf_result = torch::fmin(input1, inf_tensor);
            torch::Tensor neg_inf_tensor = torch::full_like(input1, -std::numeric_limits<float>::infinity());
            torch::Tensor neg_inf_result = torch::fmin(input1, neg_inf_tensor);
        } catch (const std::exception&) {
            // Infinity operations might behave differently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
