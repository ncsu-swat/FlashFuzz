#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        auto elements_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto test_elements_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::isin(elements_tensor, test_elements_tensor);
        
        if (offset < Size) {
            uint8_t assume_unique_byte = Data[offset++];
            bool assume_unique = (assume_unique_byte % 2) == 1;
            torch::isin(elements_tensor, test_elements_tensor, assume_unique);
        }
        
        if (offset < Size) {
            uint8_t invert_byte = Data[offset++];
            bool invert = (invert_byte % 2) == 1;
            torch::isin(elements_tensor, test_elements_tensor, false, invert);
        }
        
        if (offset < Size) {
            uint8_t assume_unique_byte = Data[offset++];
            uint8_t invert_byte = Data[offset++];
            bool assume_unique = (assume_unique_byte % 2) == 1;
            bool invert = (invert_byte % 2) == 1;
            torch::isin(elements_tensor, test_elements_tensor, assume_unique, invert);
        }
        
        torch::isin(elements_tensor, test_elements_tensor, true, true);
        torch::isin(elements_tensor, test_elements_tensor, false, false);
        
        auto scalar_tensor = torch::tensor(42);
        torch::isin(scalar_tensor, test_elements_tensor);
        torch::isin(elements_tensor, scalar_tensor);
        
        auto empty_tensor = torch::empty({0});
        torch::isin(empty_tensor, test_elements_tensor);
        torch::isin(elements_tensor, empty_tensor);
        torch::isin(empty_tensor, empty_tensor);
        
        auto large_tensor = torch::ones({1000});
        torch::isin(elements_tensor, large_tensor);
        
        if (elements_tensor.numel() > 0 && test_elements_tensor.numel() > 0) {
            auto reshaped_elements = elements_tensor.view({-1});
            auto reshaped_test = test_elements_tensor.view({-1});
            torch::isin(reshaped_elements, reshaped_test);
        }
        
        auto bool_tensor = torch::tensor({true, false, true});
        if (elements_tensor.dtype() == torch::kBool) {
            torch::isin(elements_tensor, bool_tensor);
        }
        
        if (elements_tensor.dtype() != torch::kBool && test_elements_tensor.dtype() != torch::kBool) {
            try {
                auto converted_elements = elements_tensor.to(torch::kFloat);
                auto converted_test = test_elements_tensor.to(torch::kFloat);
                torch::isin(converted_elements, converted_test);
            } catch (...) {
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}