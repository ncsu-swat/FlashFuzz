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
            torch::flatten(input_tensor);
            return 0;
        }
        
        uint8_t start_dim_byte = Data[offset++];
        int64_t start_dim = static_cast<int64_t>(static_cast<int8_t>(start_dim_byte));
        
        if (offset >= Size) {
            torch::flatten(input_tensor, start_dim);
            return 0;
        }
        
        uint8_t end_dim_byte = Data[offset++];
        int64_t end_dim = static_cast<int64_t>(static_cast<int8_t>(end_dim_byte));
        
        torch::flatten(input_tensor, start_dim, end_dim);
        
        if (offset < Size) {
            uint8_t variant_selector = Data[offset++];
            
            switch (variant_selector % 8) {
                case 0:
                    torch::flatten(input_tensor);
                    break;
                case 1:
                    torch::flatten(input_tensor, 0);
                    break;
                case 2:
                    torch::flatten(input_tensor, -1);
                    break;
                case 3:
                    torch::flatten(input_tensor, 0, -1);
                    break;
                case 4:
                    torch::flatten(input_tensor, 1, -1);
                    break;
                case 5:
                    torch::flatten(input_tensor, -2, -1);
                    break;
                case 6: {
                    int64_t large_start = 1000000;
                    torch::flatten(input_tensor, large_start);
                    break;
                }
                case 7: {
                    int64_t large_end = -1000000;
                    torch::flatten(input_tensor, 0, large_end);
                    break;
                }
            }
        }
        
        if (input_tensor.dim() == 0) {
            torch::flatten(input_tensor);
        }
        
        if (input_tensor.numel() == 0) {
            torch::flatten(input_tensor, 0, -1);
        }
        
        int64_t ndim = input_tensor.dim();
        if (ndim > 0) {
            torch::flatten(input_tensor, ndim - 1, ndim - 1);
            torch::flatten(input_tensor, 0, 0);
            torch::flatten(input_tensor, -ndim, -1);
            torch::flatten(input_tensor, ndim, ndim + 10);
            torch::flatten(input_tensor, -ndim - 10, -ndim - 1);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}