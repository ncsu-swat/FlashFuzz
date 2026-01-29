#include "fuzzer_utils.h"
#include <iostream>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.isinf operation
        torch::Tensor result = torch::isinf(input_tensor);
        
        // Test edge cases by creating special tensors with inf values
        if (offset + 1 < Size) {
            uint8_t special_case = Data[offset++] % 5;
            
            if (special_case == 0) {
                // Create tensor with inf values
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor inf_tensor = torch::full({2, 2}, std::numeric_limits<float>::infinity(), options);
                torch::Tensor inf_result = torch::isinf(inf_tensor);
            }
            else if (special_case == 1) {
                // Create tensor with -inf values
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor neg_inf_tensor = torch::full({2, 2}, -std::numeric_limits<float>::infinity(), options);
                torch::Tensor neg_inf_result = torch::isinf(neg_inf_tensor);
            }
            else if (special_case == 2) {
                // Create tensor with NaN values
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor nan_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN(), options);
                torch::Tensor nan_result = torch::isinf(nan_tensor);
            }
            else if (special_case == 3) {
                // Create tensor with mixed values (normal, inf, -inf, NaN)
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor mixed_tensor = torch::empty({2, 2}, options);
                mixed_tensor.index_put_({0, 0}, 1.0);
                mixed_tensor.index_put_({0, 1}, std::numeric_limits<float>::infinity());
                mixed_tensor.index_put_({1, 0}, -std::numeric_limits<float>::infinity());
                mixed_tensor.index_put_({1, 1}, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor mixed_result = torch::isinf(mixed_tensor);
            }
            else {
                // Test with empty tensor
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor empty_tensor = torch::empty({0}, options);
                torch::Tensor empty_result = torch::isinf(empty_tensor);
            }
        }
        
        // Test with double precision if we have more data
        if (offset + 1 < Size) {
            auto options = torch::TensorOptions().dtype(torch::kDouble);
            torch::Tensor double_tensor = torch::empty({2, 2}, options);
            double_tensor.index_put_({0, 0}, 1.0);
            double_tensor.index_put_({0, 1}, std::numeric_limits<double>::infinity());
            double_tensor.index_put_({1, 0}, -std::numeric_limits<double>::infinity());
            double_tensor.index_put_({1, 1}, std::numeric_limits<double>::quiet_NaN());
            torch::Tensor double_result = torch::isinf(double_tensor);
        }
        
        // Test with different tensor shapes
        if (offset + 2 < Size) {
            uint8_t shape_case = Data[offset++] % 3;
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            
            if (shape_case == 0) {
                // 1D tensor
                torch::Tensor t1d = torch::tensor({1.0f, std::numeric_limits<float>::infinity(), -1.0f}, options);
                torch::isinf(t1d);
            }
            else if (shape_case == 1) {
                // 3D tensor
                torch::Tensor t3d = torch::randn({2, 3, 4}, options);
                torch::isinf(t3d);
            }
            else {
                // Scalar tensor
                torch::Tensor scalar = torch::tensor(std::numeric_limits<float>::infinity(), options);
                torch::isinf(scalar);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}