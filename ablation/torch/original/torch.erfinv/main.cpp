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
        
        if (input_tensor.numel() == 0) {
            auto result = torch::erfinv(input_tensor);
            return 0;
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto result = torch::erfinv(input_tensor);
            return 0;
        }
        
        if (input_tensor.dtype() == torch::kBool || input_tensor.dtype() == torch::kInt8 || 
            input_tensor.dtype() == torch::kUInt8 || input_tensor.dtype() == torch::kInt16 || 
            input_tensor.dtype() == torch::kInt32 || input_tensor.dtype() == torch::kInt64) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        auto result = torch::erfinv(input_tensor);
        
        if (offset < Size) {
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.dtype() == input_tensor.dtype() && out_tensor.numel() >= result.numel()) {
                torch::erfinv_out(out_tensor, input_tensor);
            }
        }
        
        auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
        auto inf_result = torch::erfinv(inf_tensor);
        
        auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<float>::infinity());
        auto neg_inf_result = torch::erfinv(neg_inf_tensor);
        
        auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
        auto nan_result = torch::erfinv(nan_tensor);
        
        auto ones_tensor = torch::ones_like(input_tensor);
        auto ones_result = torch::erfinv(ones_tensor);
        
        auto neg_ones_tensor = torch::full_like(input_tensor, -1.0);
        auto neg_ones_result = torch::erfinv(neg_ones_tensor);
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_result = torch::erfinv(zero_tensor);
        
        auto large_tensor = torch::full_like(input_tensor, 0.999999);
        auto large_result = torch::erfinv(large_tensor);
        
        auto small_tensor = torch::full_like(input_tensor, 1e-10);
        auto small_result = torch::erfinv(small_tensor);
        
        auto boundary_tensor = torch::full_like(input_tensor, 1.0001);
        auto boundary_result = torch::erfinv(boundary_tensor);
        
        auto neg_boundary_tensor = torch::full_like(input_tensor, -1.0001);
        auto neg_boundary_result = torch::erfinv(neg_boundary_tensor);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}