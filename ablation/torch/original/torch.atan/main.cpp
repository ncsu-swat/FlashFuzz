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
            uint8_t out_selector = Data[offset++];
            if (out_selector % 2 == 1 && offset < Size) {
                auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (out_tensor.sizes() == input_tensor.sizes()) {
                    torch::atan_out(out_tensor, input_tensor);
                }
            }
        }
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::atan(input_tensor2);
        }
        
        if (input_tensor.numel() > 0) {
            auto cloned = input_tensor.clone();
            torch::atan_(cloned);
        }
        
        if (offset < Size && input_tensor.numel() > 0) {
            uint8_t dtype_selector = Data[offset++];
            auto target_dtype = fuzzer_utils::parseDataType(dtype_selector);
            try {
                auto converted = input_tensor.to(target_dtype);
                torch::atan(converted);
            } catch (...) {
            }
        }
        
        if (input_tensor.dim() > 0 && input_tensor.numel() > 0) {
            try {
                auto reshaped = input_tensor.view({-1});
                torch::atan(reshaped);
            } catch (...) {
            }
        }
        
        if (input_tensor.numel() > 1) {
            try {
                auto sliced = input_tensor.slice(0, 0, 1);
                torch::atan(sliced);
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