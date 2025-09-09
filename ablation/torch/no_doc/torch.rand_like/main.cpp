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
        
        auto result = torch::rand_like(input_tensor);
        
        if (offset < Size) {
            uint8_t options_byte = Data[offset++];
            
            if (options_byte & 0x01) {
                auto dtype_selector = (offset < Size) ? Data[offset++] : 0;
                auto target_dtype = fuzzer_utils::parseDataType(dtype_selector);
                result = torch::rand_like(input_tensor, torch::TensorOptions().dtype(target_dtype));
            }
            
            if (options_byte & 0x02) {
                result = torch::rand_like(input_tensor, torch::TensorOptions().layout(torch::kStrided));
            }
            
            if (options_byte & 0x04) {
                result = torch::rand_like(input_tensor, torch::TensorOptions().device(torch::kCPU));
            }
            
            if (options_byte & 0x08) {
                result = torch::rand_like(input_tensor, torch::TensorOptions().requires_grad(true));
            }
            
            if (options_byte & 0x10) {
                result = torch::rand_like(input_tensor, torch::TensorOptions().requires_grad(false));
            }
            
            if (options_byte & 0x20) {
                result = torch::rand_like(input_tensor, torch::TensorOptions().pinned_memory(false));
            }
            
            if (options_byte & 0x40) {
                auto memory_format = torch::MemoryFormat::Contiguous;
                if (offset < Size) {
                    uint8_t format_byte = Data[offset++];
                    switch (format_byte % 4) {
                        case 0: memory_format = torch::MemoryFormat::Contiguous; break;
                        case 1: memory_format = torch::MemoryFormat::Preserve; break;
                        case 2: memory_format = torch::MemoryFormat::ChannelsLast; break;
                        case 3: memory_format = torch::MemoryFormat::ChannelsLast3d; break;
                    }
                }
                result = torch::rand_like(input_tensor, torch::TensorOptions(), memory_format);
            }
            
            if (options_byte & 0x80) {
                if (offset + 1 < Size) {
                    auto dtype_selector = Data[offset++];
                    auto target_dtype = fuzzer_utils::parseDataType(dtype_selector);
                    auto format_byte = Data[offset++];
                    auto memory_format = torch::MemoryFormat::Contiguous;
                    switch (format_byte % 4) {
                        case 0: memory_format = torch::MemoryFormat::Contiguous; break;
                        case 1: memory_format = torch::MemoryFormat::Preserve; break;
                        case 2: memory_format = torch::MemoryFormat::ChannelsLast; break;
                        case 3: memory_format = torch::MemoryFormat::ChannelsLast3d; break;
                    }
                    result = torch::rand_like(input_tensor, torch::TensorOptions().dtype(target_dtype), memory_format);
                }
            }
        }
        
        if (offset < Size) {
            auto second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                auto result2 = torch::rand_like(second_tensor);
                auto combined = result + result2;
            } catch (...) {
            }
        }
        
        result.sum();
        result.mean();
        
        if (result.numel() > 0) {
            result.item();
        }
        
        auto cloned = result.clone();
        auto detached = result.detach();
        
        if (input_tensor.is_floating_point()) {
            result.sin();
            result.cos();
        }
        
        if (input_tensor.dtype() == torch::kBool) {
            result.logical_not();
        }
        
        result.to(torch::kFloat);
        result.to(torch::kCPU);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}