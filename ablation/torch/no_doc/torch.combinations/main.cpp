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
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t r_byte = Data[offset++];
        int64_t r = static_cast<int64_t>(r_byte % 10);
        
        if (offset < Size) {
            uint8_t with_replacement_byte = Data[offset++];
            bool with_replacement = (with_replacement_byte % 2) == 1;
            
            torch::combinations(input_tensor, r, with_replacement);
        } else {
            torch::combinations(input_tensor, r);
        }
        
        if (input_tensor.numel() > 0 && input_tensor.dim() == 1) {
            torch::combinations(input_tensor, 0);
            torch::combinations(input_tensor, 1);
            torch::combinations(input_tensor, input_tensor.size(0));
            torch::combinations(input_tensor, input_tensor.size(0) + 1);
            torch::combinations(input_tensor, -1);
            
            torch::combinations(input_tensor, 0, true);
            torch::combinations(input_tensor, 1, true);
            torch::combinations(input_tensor, input_tensor.size(0), true);
            torch::combinations(input_tensor, input_tensor.size(0) + 1, true);
            torch::combinations(input_tensor, -1, true);
            
            torch::combinations(input_tensor, 0, false);
            torch::combinations(input_tensor, 1, false);
            torch::combinations(input_tensor, input_tensor.size(0), false);
            torch::combinations(input_tensor, input_tensor.size(0) + 1, false);
            torch::combinations(input_tensor, -1, false);
        }
        
        if (input_tensor.numel() == 0) {
            torch::combinations(input_tensor, 0);
            torch::combinations(input_tensor, 1);
            torch::combinations(input_tensor, -1);
            torch::combinations(input_tensor, 0, true);
            torch::combinations(input_tensor, 1, true);
            torch::combinations(input_tensor, -1, true);
            torch::combinations(input_tensor, 0, false);
            torch::combinations(input_tensor, 1, false);
            torch::combinations(input_tensor, -1, false);
        }
        
        if (input_tensor.dim() > 1) {
            torch::combinations(input_tensor, 0);
            torch::combinations(input_tensor, 1);
            torch::combinations(input_tensor, -1);
            torch::combinations(input_tensor, 0, true);
            torch::combinations(input_tensor, 1, true);
            torch::combinations(input_tensor, -1, true);
            torch::combinations(input_tensor, 0, false);
            torch::combinations(input_tensor, 1, false);
            torch::combinations(input_tensor, -1, false);
        }
        
        auto scalar_tensor = torch::tensor(42);
        torch::combinations(scalar_tensor, 0);
        torch::combinations(scalar_tensor, 1);
        torch::combinations(scalar_tensor, -1);
        torch::combinations(scalar_tensor, 0, true);
        torch::combinations(scalar_tensor, 1, true);
        torch::combinations(scalar_tensor, -1, true);
        torch::combinations(scalar_tensor, 0, false);
        torch::combinations(scalar_tensor, 1, false);
        torch::combinations(scalar_tensor, -1, false);
        
        auto empty_tensor = torch::empty({0});
        torch::combinations(empty_tensor, 0);
        torch::combinations(empty_tensor, 1);
        torch::combinations(empty_tensor, -1);
        torch::combinations(empty_tensor, 0, true);
        torch::combinations(empty_tensor, 1, true);
        torch::combinations(empty_tensor, -1, true);
        torch::combinations(empty_tensor, 0, false);
        torch::combinations(empty_tensor, 1, false);
        torch::combinations(empty_tensor, -1, false);
        
        auto large_r_tensor = torch::arange(5);
        torch::combinations(large_r_tensor, 100);
        torch::combinations(large_r_tensor, 1000);
        torch::combinations(large_r_tensor, 100, true);
        torch::combinations(large_r_tensor, 1000, true);
        torch::combinations(large_r_tensor, 100, false);
        torch::combinations(large_r_tensor, 1000, false);
        
        if (offset < Size) {
            int64_t extreme_r = static_cast<int64_t>(Data[offset]);
            extreme_r = (extreme_r << 56) | (extreme_r << 48) | (extreme_r << 40) | (extreme_r << 32) | 
                       (extreme_r << 24) | (extreme_r << 16) | (extreme_r << 8) | extreme_r;
            torch::combinations(input_tensor, extreme_r);
            torch::combinations(input_tensor, extreme_r, true);
            torch::combinations(input_tensor, extreme_r, false);
            torch::combinations(input_tensor, -extreme_r);
            torch::combinations(input_tensor, -extreme_r, true);
            torch::combinations(input_tensor, -extreme_r, false);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}