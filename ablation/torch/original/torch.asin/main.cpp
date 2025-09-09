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
            auto out_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (out_tensor.sizes() == result.sizes() && out_tensor.dtype() == result.dtype()) {
                torch::asin_out(out_tensor, input_tensor);
            }
        }
        
        auto input_copy = input_tensor.clone();
        input_copy.asin_();
        
        if (offset + 1 < Size) {
            uint8_t test_selector = Data[offset];
            
            if (test_selector % 4 == 0) {
                auto empty_tensor = torch::empty({0});
                torch::asin(empty_tensor);
            } else if (test_selector % 4 == 1) {
                auto scalar_tensor = torch::tensor(0.5);
                torch::asin(scalar_tensor);
            } else if (test_selector % 4 == 2) {
                auto large_tensor = torch::ones({1000, 1000}) * 0.1;
                torch::asin(large_tensor);
            } else {
                auto edge_values = torch::tensor({-1.0, -0.9999, 0.0, 0.9999, 1.0});
                torch::asin(edge_values);
            }
        }
        
        auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::infinity());
        torch::asin(inf_tensor);
        
        auto neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<double>::infinity());
        torch::asin(neg_inf_tensor);
        
        auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<double>::quiet_NaN());
        torch::asin(nan_tensor);
        
        if (input_tensor.numel() > 0) {
            auto extreme_values = torch::tensor({-2.0, 2.0, 10.0, -10.0});
            torch::asin(extreme_values);
        }
        
        auto complex_tensor = torch::complex(input_tensor, input_tensor);
        torch::asin(complex_tensor);
        
        if (input_tensor.dim() > 0) {
            auto reshaped = input_tensor.view(-1);
            torch::asin(reshaped);
        }
        
        auto detached = input_tensor.detach();
        torch::asin(detached);
        
        if (input_tensor.dtype() != torch::kBool) {
            auto bool_tensor = input_tensor > 0;
            auto bool_float = bool_tensor.to(torch::kFloat);
            torch::asin(bool_float);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}