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
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            torch::nanmedian(input_tensor);
            return 0;
        }
        
        uint8_t operation_selector = Data[offset++];
        
        if (operation_selector % 2 == 0) {
            torch::nanmedian(input_tensor);
        } else {
            if (offset >= Size) {
                torch::nanmedian(input_tensor);
                return 0;
            }
            
            uint8_t dim_selector = Data[offset++];
            int64_t dim = static_cast<int64_t>(dim_selector) - 128;
            
            if (offset < Size) {
                uint8_t keepdim_selector = Data[offset++];
                bool keepdim = (keepdim_selector % 2) == 1;
                torch::nanmedian(input_tensor, dim, keepdim);
            } else {
                torch::nanmedian(input_tensor, dim);
            }
        }
        
        if (input_tensor.numel() > 0 && input_tensor.dim() > 0) {
            for (int64_t d = -input_tensor.dim(); d < input_tensor.dim(); ++d) {
                torch::nanmedian(input_tensor, d);
                torch::nanmedian(input_tensor, d, true);
                torch::nanmedian(input_tensor, d, false);
            }
        }
        
        if (input_tensor.dim() > 1) {
            torch::nanmedian(input_tensor, input_tensor.dim() + 10);
            torch::nanmedian(input_tensor, -input_tensor.dim() - 10);
        }
        
        auto empty_tensor = torch::empty({0}, input_tensor.options());
        torch::nanmedian(empty_tensor);
        
        auto single_element = torch::tensor({std::numeric_limits<double>::quiet_NaN()}, input_tensor.options().dtype(torch::kFloat64));
        torch::nanmedian(single_element);
        
        auto all_nan = torch::full({3, 3}, std::numeric_limits<double>::quiet_NaN(), input_tensor.options().dtype(torch::kFloat64));
        torch::nanmedian(all_nan);
        torch::nanmedian(all_nan, 0);
        torch::nanmedian(all_nan, 1);
        
        if (input_tensor.dtype() == torch::kFloat32 || input_tensor.dtype() == torch::kFloat64) {
            auto mixed_tensor = input_tensor.clone();
            if (mixed_tensor.numel() > 0) {
                mixed_tensor.view(-1)[0] = std::numeric_limits<double>::quiet_NaN();
                torch::nanmedian(mixed_tensor);
                if (mixed_tensor.dim() > 0) {
                    torch::nanmedian(mixed_tensor, 0);
                }
            }
        }
        
        auto large_tensor = torch::randn({1000}, input_tensor.options().dtype(torch::kFloat32));
        torch::nanmedian(large_tensor);
        
        if (input_tensor.numel() > 1) {
            auto reshaped = input_tensor.view(-1);
            torch::nanmedian(reshaped);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}