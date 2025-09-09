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
        
        if (offset >= Size) {
            auto result = torch::empty_like(input_tensor);
            return 0;
        }

        uint8_t options_byte = Data[offset++];
        
        torch::TensorOptions options;
        bool has_dtype = (options_byte & 0x01) != 0;
        bool has_device = (options_byte & 0x02) != 0;
        bool has_layout = (options_byte & 0x04) != 0;
        bool has_requires_grad = (options_byte & 0x08) != 0;
        bool has_memory_format = (options_byte & 0x10) != 0;

        if (has_dtype && offset < Size) {
            auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
            options = options.dtype(dtype);
        }

        if (has_device && offset < Size) {
            uint8_t device_selector = Data[offset++];
            if (device_selector % 2 == 0) {
                options = options.device(torch::kCPU);
            } else {
                if (torch::cuda::is_available()) {
                    options = options.device(torch::kCUDA);
                } else {
                    options = options.device(torch::kCPU);
                }
            }
        }

        if (has_layout && offset < Size) {
            uint8_t layout_selector = Data[offset++];
            if (layout_selector % 2 == 0) {
                options = options.layout(torch::kStrided);
            } else {
                options = options.layout(torch::kSparse);
            }
        }

        if (has_requires_grad && offset < Size) {
            uint8_t grad_selector = Data[offset++];
            options = options.requires_grad(grad_selector % 2 == 1);
        }

        torch::MemoryFormat memory_format = torch::MemoryFormat::Preserve;
        if (has_memory_format && offset < Size) {
            uint8_t format_selector = Data[offset++];
            switch (format_selector % 4) {
                case 0:
                    memory_format = torch::MemoryFormat::Preserve;
                    break;
                case 1:
                    memory_format = torch::MemoryFormat::Contiguous;
                    break;
                case 2:
                    memory_format = torch::MemoryFormat::ChannelsLast;
                    break;
                case 3:
                    memory_format = torch::MemoryFormat::ChannelsLast3d;
                    break;
            }
        }

        torch::Tensor result;
        if (has_dtype || has_device || has_layout || has_requires_grad) {
            if (has_memory_format) {
                result = torch::empty_like(input_tensor, options, memory_format);
            } else {
                result = torch::empty_like(input_tensor, options);
            }
        } else {
            if (has_memory_format) {
                result = torch::empty_like(input_tensor, torch::TensorOptions(), memory_format);
            } else {
                result = torch::empty_like(input_tensor);
            }
        }

        if (result.numel() > 0) {
            auto sum = result.sum();
        }

        auto cloned = result.clone();
        auto detached = result.detach();

        if (result.dim() > 0) {
            auto reshaped = result.reshape({-1});
        }

        if (result.numel() > 1) {
            auto sliced = result.slice(0, 0, std::min(static_cast<int64_t>(2), result.size(0)));
        }

        auto moved = std::move(result);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}