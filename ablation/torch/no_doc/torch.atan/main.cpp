#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        auto result = torch::atan(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            auto result2 = torch::atan(input_tensor2);
        }
        
        auto scalar_input = torch::scalar_tensor(3.14159, torch::kFloat);
        auto scalar_result = torch::atan(scalar_input);
        
        auto complex_input = torch::tensor({{1.0, 2.0}, {-1.0, -2.0}}, torch::kComplexFloat);
        auto complex_result = torch::atan(complex_input);
        
        auto large_input = torch::tensor({1e10, -1e10, 1e-10, -1e-10}, torch::kDouble);
        auto large_result = torch::atan(large_input);
        
        auto inf_input = torch::tensor({std::numeric_limits<float>::infinity(), 
                                       -std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::quiet_NaN()}, torch::kFloat);
        auto inf_result = torch::atan(inf_input);
        
        auto zero_input = torch::zeros({3, 3}, torch::kFloat);
        auto zero_result = torch::atan(zero_input);
        
        auto empty_input = torch::empty({0}, torch::kFloat);
        auto empty_result = torch::atan(empty_input);
        
        auto high_dim_input = torch::randn({2, 3, 4, 5}, torch::kFloat);
        auto high_dim_result = torch::atan(high_dim_input);
        
        if (input_tensor.numel() > 0) {
            auto inplace_tensor = input_tensor.clone();
            inplace_tensor.atan_();
        }
        
        auto bool_input = torch::tensor({true, false}, torch::kBool);
        auto bool_result = torch::atan(bool_input);
        
        auto int_input = torch::tensor({-5, 0, 5}, torch::kInt32);
        auto int_result = torch::atan(int_input);
        
        auto half_input = torch::tensor({1.5, -1.5}, torch::kHalf);
        auto half_result = torch::atan(half_input);
        
        auto bfloat16_input = torch::tensor({2.5, -2.5}, torch::kBFloat16);
        auto bfloat16_result = torch::atan(bfloat16_input);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}