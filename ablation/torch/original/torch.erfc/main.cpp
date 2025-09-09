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
        
        auto result = torch::erfc(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor out_tensor;
            try {
                out_tensor = torch::empty_like(input_tensor2);
                torch::erfc_out(out_tensor, input_tensor2);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            auto input_tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor3.dtype() == torch::kFloat || 
                input_tensor3.dtype() == torch::kDouble ||
                input_tensor3.dtype() == torch::kHalf ||
                input_tensor3.dtype() == torch::kBFloat16) {
                
                auto inf_tensor = torch::full_like(input_tensor3, std::numeric_limits<double>::infinity());
                auto neg_inf_tensor = torch::full_like(input_tensor3, -std::numeric_limits<double>::infinity());
                auto nan_tensor = torch::full_like(input_tensor3, std::numeric_limits<double>::quiet_NaN());
                auto zero_tensor = torch::zeros_like(input_tensor3);
                auto large_pos_tensor = torch::full_like(input_tensor3, 1e10);
                auto large_neg_tensor = torch::full_like(input_tensor3, -1e10);
                
                torch::erfc(inf_tensor);
                torch::erfc(neg_inf_tensor);
                torch::erfc(nan_tensor);
                torch::erfc(zero_tensor);
                torch::erfc(large_pos_tensor);
                torch::erfc(large_neg_tensor);
            }
        }
        
        if (offset < Size) {
            auto input_tensor4 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor4.numel() > 0) {
                auto reshaped = input_tensor4.view({-1});
                torch::erfc(reshaped);
                
                if (input_tensor4.dim() > 1) {
                    auto transposed = input_tensor4.transpose(0, -1);
                    torch::erfc(transposed);
                }
                
                auto contiguous = input_tensor4.contiguous();
                torch::erfc(contiguous);
            }
        }
        
        if (offset < Size) {
            auto input_tensor5 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor5.dtype() == torch::kComplexFloat || 
                input_tensor5.dtype() == torch::kComplexDouble) {
                torch::erfc(input_tensor5);
            }
        }
        
        if (input_tensor.numel() == 0) {
            torch::erfc(input_tensor);
        }
        
        if (input_tensor.dim() == 0) {
            torch::erfc(input_tensor);
        }
        
        auto cloned = input_tensor.clone();
        torch::erfc(cloned);
        
        if (input_tensor.is_cuda() == false) {
            auto cpu_result = torch::erfc(input_tensor.cpu());
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}