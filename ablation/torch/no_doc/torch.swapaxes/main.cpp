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
        
        auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t axis0_byte = Data[offset++];
        int64_t axis0 = static_cast<int64_t>(static_cast<int8_t>(axis0_byte));
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t axis1_byte = Data[offset++];
        int64_t axis1 = static_cast<int64_t>(static_cast<int8_t>(axis1_byte));
        
        auto result = torch::swapaxes(tensor, axis0, axis1);
        
        if (offset < Size) {
            uint8_t negative_test = Data[offset++];
            if (negative_test % 4 == 0) {
                int64_t large_axis = static_cast<int64_t>(negative_test) * 1000;
                torch::swapaxes(tensor, large_axis, axis1);
            } else if (negative_test % 4 == 1) {
                int64_t large_axis = static_cast<int64_t>(negative_test) * 1000;
                torch::swapaxes(tensor, axis0, large_axis);
            } else if (negative_test % 4 == 2) {
                int64_t very_negative = -static_cast<int64_t>(negative_test) - 1000;
                torch::swapaxes(tensor, very_negative, axis1);
            } else {
                int64_t very_negative = -static_cast<int64_t>(negative_test) - 1000;
                torch::swapaxes(tensor, axis0, very_negative);
            }
        }
        
        if (tensor.dim() > 0) {
            torch::swapaxes(tensor, 0, tensor.dim() - 1);
            torch::swapaxes(tensor, -1, -tensor.dim());
            torch::swapaxes(tensor, tensor.dim(), 0);
            torch::swapaxes(tensor, 0, -tensor.dim() - 1);
        }
        
        torch::swapaxes(tensor, 0, 0);
        torch::swapaxes(tensor, -1, -1);
        
        if (tensor.dim() >= 2) {
            for (int64_t i = 0; i < tensor.dim(); ++i) {
                for (int64_t j = i + 1; j < tensor.dim(); ++j) {
                    torch::swapaxes(tensor, i, j);
                    torch::swapaxes(tensor, -i - 1, -j - 1);
                }
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