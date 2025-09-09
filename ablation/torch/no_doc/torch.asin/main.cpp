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
        
        auto result = torch::asin(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            auto result2 = input_tensor2.asin();
        }
        
        if (offset < Size) {
            auto input_tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::asin_(input_tensor3);
        }
        
        if (offset < Size) {
            auto input_tensor4 = fuzzer_utils::createTensor(Data, Size, offset);
            auto output_tensor = torch::empty_like(input_tensor4);
            torch::asin_out(output_tensor, input_tensor4);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}