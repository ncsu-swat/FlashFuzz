#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        uint8_t config_byte = Data[offset++];
        bool assume_unique = (config_byte & 0x01) != 0;
        bool invert = (config_byte & 0x02) != 0;
        bool elements_is_scalar = (config_byte & 0x04) != 0;
        bool test_elements_is_scalar = (config_byte & 0x08) != 0;
        
        if (elements_is_scalar && test_elements_is_scalar) {
            elements_is_scalar = false;
        }
        
        torch::Tensor elements;
        torch::Tensor test_elements;
        torch::Scalar elements_scalar;
        torch::Scalar test_elements_scalar;
        
        if (elements_is_scalar) {
            if (offset + sizeof(double) <= Size) {
                double scalar_val;
                std::memcpy(&scalar_val, Data + offset, sizeof(double));
                offset += sizeof(double);
                elements_scalar = torch::Scalar(scalar_val);
            } else {
                elements_scalar = torch::Scalar(0.0);
            }
            test_elements = fuzzer_utils::createTensor(Data, Size, offset);
        } else if (test_elements_is_scalar) {
            elements = fuzzer_utils::createTensor(Data, Size, offset);
            if (offset + sizeof(double) <= Size) {
                double scalar_val;
                std::memcpy(&scalar_val, Data + offset, sizeof(double));
                offset += sizeof(double);
                test_elements_scalar = torch::Scalar(scalar_val);
            } else {
                test_elements_scalar = torch::Scalar(0.0);
            }
        } else {
            elements = fuzzer_utils::createTensor(Data, Size, offset);
            test_elements = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        torch::Tensor result;
        
        if (elements_is_scalar) {
            result = torch::isin(elements_scalar, test_elements, assume_unique, invert);
        } else if (test_elements_is_scalar) {
            result = torch::isin(elements, test_elements_scalar, assume_unique, invert);
        } else {
            result = torch::isin(elements, test_elements, assume_unique, invert);
        }
        
        if (result.numel() > 0) {
            auto first_val = result.flatten()[0].item<bool>();
            (void)first_val;
        }
        
        if (result.dim() > 0 && result.size(0) > 0) {
            auto sum_result = torch::sum(result);
            (void)sum_result;
        }
        
        if (!elements_is_scalar && elements.numel() > 0) {
            auto empty_test = torch::empty({0}, elements.options());
            auto empty_result = torch::isin(elements, empty_test, assume_unique, invert);
            (void)empty_result;
        }
        
        if (!test_elements_is_scalar && test_elements.numel() > 0) {
            auto empty_elements = torch::empty({0}, test_elements.options());
            auto empty_result2 = torch::isin(empty_elements, test_elements, assume_unique, invert);
            (void)empty_result2;
        }
        
        if (!elements_is_scalar && !test_elements_is_scalar && 
            elements.numel() > 0 && test_elements.numel() > 0) {
            auto both_empty = torch::isin(torch::empty({0}, elements.options()), 
                                        torch::empty({0}, test_elements.options()), 
                                        assume_unique, invert);
            (void)both_empty;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}