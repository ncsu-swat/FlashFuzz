#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 1) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        auto result1 = torch::rand_like(input_tensor);
        
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto target_dtype = fuzzer_utils::parseDataType(dtype_selector);
            auto result2 = torch::rand_like(input_tensor, torch::TensorOptions().dtype(target_dtype));
        }
        
        if (offset < Size) {
            uint8_t requires_grad_flag = Data[offset++];
            bool requires_grad = (requires_grad_flag % 2) == 1;
            auto result3 = torch::rand_like(input_tensor, torch::TensorOptions().requires_grad(requires_grad));
        }
        
        if (offset < Size) {
            uint8_t layout_selector = Data[offset++];
            torch::Layout target_layout = (layout_selector % 2 == 0) ? torch::kStrided : torch::kSparse;
            if (target_layout == torch::kStrided) {
                auto result4 = torch::rand_like(input_tensor, torch::TensorOptions().layout(target_layout));
            }
        }
        
        if (offset < Size) {
            uint8_t memory_format_selector = Data[offset++];
            torch::MemoryFormat memory_format;
            switch (memory_format_selector % 4) {
                case 0: memory_format = torch::MemoryFormat::Contiguous; break;
                case 1: memory_format = torch::MemoryFormat::Preserve; break;
                case 2: memory_format = torch::MemoryFormat::ChannelsLast; break;
                default: memory_format = torch::MemoryFormat::ChannelsLast3d; break;
            }
            auto result5 = torch::rand_like(input_tensor, torch::TensorOptions().memory_format(memory_format));
        }
        
        if (offset + 2 < Size) {
            uint8_t dtype_selector = Data[offset++];
            uint8_t requires_grad_flag = Data[offset++];
            auto target_dtype = fuzzer_utils::parseDataType(dtype_selector);
            bool requires_grad = (requires_grad_flag % 2) == 1;
            auto result6 = torch::rand_like(input_tensor, 
                torch::TensorOptions().dtype(target_dtype).requires_grad(requires_grad));
        }
        
        if (input_tensor.numel() == 0) {
            auto empty_result = torch::rand_like(input_tensor);
        }
        
        if (input_tensor.dim() == 0) {
            auto scalar_result = torch::rand_like(input_tensor);
        }
        
        if (input_tensor.dtype() == torch::kBool) {
            auto bool_result = torch::rand_like(input_tensor);
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_result = torch::rand_like(input_tensor);
        }
        
        auto detached_input = input_tensor.detach();
        auto detached_result = torch::rand_like(detached_input);
        
        if (input_tensor.is_contiguous()) {
            auto contiguous_result = torch::rand_like(input_tensor);
        }
        
        auto cloned_input = input_tensor.clone();
        auto cloned_result = torch::rand_like(cloned_input);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}