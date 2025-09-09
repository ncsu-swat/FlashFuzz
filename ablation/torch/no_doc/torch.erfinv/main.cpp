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
        
        if (input_tensor.dtype() == torch::kBool || 
            input_tensor.dtype() == torch::kInt8 || 
            input_tensor.dtype() == torch::kUInt8 || 
            input_tensor.dtype() == torch::kInt16 || 
            input_tensor.dtype() == torch::kInt32 || 
            input_tensor.dtype() == torch::kInt64) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || 
            input_tensor.dtype() == torch::kComplexDouble) {
            input_tensor = torch::real(input_tensor);
        }
        
        auto result = torch::erfinv(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor2.numel() > 0) {
                if (input_tensor2.dtype() == torch::kBool || 
                    input_tensor2.dtype() == torch::kInt8 || 
                    input_tensor2.dtype() == torch::kUInt8 || 
                    input_tensor2.dtype() == torch::kInt16 || 
                    input_tensor2.dtype() == torch::kInt32 || 
                    input_tensor2.dtype() == torch::kInt64) {
                    input_tensor2 = input_tensor2.to(torch::kFloat);
                }
                
                if (input_tensor2.dtype() == torch::kComplexFloat || 
                    input_tensor2.dtype() == torch::kComplexDouble) {
                    input_tensor2 = torch::real(input_tensor2);
                }
                
                auto result2 = torch::erfinv(input_tensor2);
            }
        }
        
        auto scalar_tensor = torch::tensor(0.5);
        auto scalar_result = torch::erfinv(scalar_tensor);
        
        auto edge_values = torch::tensor({-0.999999, -0.5, 0.0, 0.5, 0.999999});
        auto edge_result = torch::erfinv(edge_values);
        
        auto extreme_values = torch::tensor({-1.0, 1.0});
        auto extreme_result = torch::erfinv(extreme_values);
        
        if (input_tensor.numel() > 0 && input_tensor.dim() > 0) {
            auto flattened = input_tensor.flatten();
            auto flat_result = torch::erfinv(flattened);
            
            if (input_tensor.dim() > 1) {
                auto reshaped = input_tensor.view({-1});
                auto reshape_result = torch::erfinv(reshaped);
            }
        }
        
        auto large_tensor = torch::randn({100, 100}) * 0.9;
        auto large_result = torch::erfinv(large_tensor);
        
        auto tiny_tensor = torch::tensor({1e-10, -1e-10});
        auto tiny_result = torch::erfinv(tiny_tensor);
        
        auto boundary_tensor = torch::tensor({-0.9999999999, 0.9999999999});
        auto boundary_result = torch::erfinv(boundary_tensor);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}