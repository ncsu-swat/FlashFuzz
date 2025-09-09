#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 1) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input_tensor.dim() < 2) {
            auto reshaped = input_tensor.view({1, 1});
            torch::trace(reshaped);
        } else {
            torch::trace(input_tensor);
        }
        
        if (offset < Size) {
            auto second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (second_tensor.dim() >= 2) {
                torch::trace(second_tensor);
            }
        }
        
        auto zero_tensor = torch::zeros({0, 0});
        torch::trace(zero_tensor);
        
        auto large_tensor = torch::randn({1000, 1000});
        torch::trace(large_tensor);
        
        auto complex_tensor = torch::randn({3, 3}, torch::dtype(torch::kComplexFloat));
        torch::trace(complex_tensor);
        
        auto bool_tensor = torch::randint(0, 2, {2, 2}, torch::dtype(torch::kBool));
        torch::trace(bool_tensor);
        
        auto int_tensor = torch::randint(-100, 100, {5, 5}, torch::dtype(torch::kInt64));
        torch::trace(int_tensor);
        
        auto single_element = torch::tensor({{42.0}});
        torch::trace(single_element);
        
        auto rectangular = torch::randn({2, 5});
        torch::trace(rectangular);
        
        auto tall_rectangular = torch::randn({10, 3});
        torch::trace(tall_rectangular);
        
        if (input_tensor.numel() > 0 && input_tensor.dim() >= 1) {
            auto flattened = input_tensor.flatten();
            if (flattened.numel() >= 4) {
                auto square_size = static_cast<int64_t>(std::sqrt(flattened.numel()));
                if (square_size > 0) {
                    auto square_tensor = flattened.narrow(0, 0, square_size * square_size).view({square_size, square_size});
                    torch::trace(square_tensor);
                }
            }
        }
        
        auto inf_tensor = torch::full({3, 3}, std::numeric_limits<float>::infinity());
        torch::trace(inf_tensor);
        
        auto nan_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN());
        torch::trace(nan_tensor);
        
        auto very_small = torch::full({4, 4}, std::numeric_limits<float>::min());
        torch::trace(very_small);
        
        auto very_large = torch::full({3, 3}, std::numeric_limits<float>::max());
        torch::trace(very_large);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}