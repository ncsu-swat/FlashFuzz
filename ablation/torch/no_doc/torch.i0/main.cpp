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
        
        if (input_tensor.dtype() == torch::kBool || 
            input_tensor.dtype() == torch::kInt8 ||
            input_tensor.dtype() == torch::kUInt8 ||
            input_tensor.dtype() == torch::kInt16 ||
            input_tensor.dtype() == torch::kInt32 ||
            input_tensor.dtype() == torch::kInt64) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        auto result = torch::i0(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input_tensor2.dtype() == torch::kBool || 
                input_tensor2.dtype() == torch::kInt8 ||
                input_tensor2.dtype() == torch::kUInt8 ||
                input_tensor2.dtype() == torch::kInt16 ||
                input_tensor2.dtype() == torch::kInt32 ||
                input_tensor2.dtype() == torch::kInt64) {
                input_tensor2 = input_tensor2.to(torch::kFloat);
            }
            
            auto result2 = torch::i0(input_tensor2);
        }
        
        if (input_tensor.numel() > 0) {
            auto scalar_input = input_tensor.flatten()[0];
            auto scalar_result = torch::i0(scalar_input);
        }
        
        if (input_tensor.dim() > 0) {
            auto reshaped = input_tensor.view({-1});
            auto reshaped_result = torch::i0(reshaped);
        }
        
        auto cloned_input = input_tensor.clone();
        auto cloned_result = torch::i0(cloned_input);
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto negative_input = -input_tensor;
            auto negative_result = torch::i0(negative_input);
            
            auto large_input = input_tensor * 1000.0;
            auto large_result = torch::i0(large_input);
            
            auto small_input = input_tensor * 0.001;
            auto small_result = torch::i0(small_input);
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_result = torch::i0(input_tensor);
        }
        
        auto contiguous_input = input_tensor.contiguous();
        auto contiguous_result = torch::i0(contiguous_input);
        
        if (input_tensor.dim() > 1) {
            auto transposed = input_tensor.transpose(0, 1);
            auto transposed_result = torch::i0(transposed);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}