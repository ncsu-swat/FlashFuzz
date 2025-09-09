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

        uint8_t num_dims_byte = Data[offset++];
        uint8_t num_dims = (num_dims_byte % 5);
        
        std::vector<int64_t> shape;
        
        for (uint8_t i = 0; i < num_dims; ++i) {
            if (offset + sizeof(int64_t) <= Size) {
                int64_t dim_raw;
                std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                int64_t dim = dim_raw;
                shape.push_back(dim);
            } else {
                int64_t fallback_dim = (offset < Size) ? static_cast<int64_t>(Data[offset++]) : 1;
                shape.push_back(fallback_dim);
            }
        }

        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }

        bool requires_grad = false;
        if (offset < Size) {
            requires_grad = (Data[offset++] % 2) == 1;
        }

        torch::Device device = torch::kCPU;
        if (offset < Size) {
            uint8_t device_selector = Data[offset++];
            if (device_selector % 2 == 1 && torch::cuda::is_available()) {
                device = torch::kCUDA;
            }
        }

        torch::Layout layout = torch::kStrided;
        if (offset < Size) {
            uint8_t layout_selector = Data[offset++];
            if (layout_selector % 2 == 1) {
                layout = torch::kSparse;
            }
        }

        bool pin_memory = false;
        if (offset < Size) {
            pin_memory = (Data[offset++] % 2) == 1;
        }

        torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous;
        if (offset < Size) {
            uint8_t format_selector = Data[offset++];
            switch (format_selector % 4) {
                case 0: memory_format = torch::MemoryFormat::Contiguous; break;
                case 1: memory_format = torch::MemoryFormat::Preserve; break;
                case 2: memory_format = torch::MemoryFormat::ChannelsLast; break;
                case 3: memory_format = torch::MemoryFormat::ChannelsLast3d; break;
            }
        }

        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .layout(layout)
            .requires_grad(requires_grad)
            .pinned_memory(pin_memory);

        torch::Tensor result;
        
        if (shape.empty()) {
            result = torch::empty({}, options, memory_format);
        } else {
            result = torch::empty(shape, options, memory_format);
        }

        if (result.numel() > 0) {
            auto sum = result.sum();
            auto mean_val = result.mean();
            auto std_val = result.std();
        }

        auto reshaped = result.view({-1});
        auto cloned = result.clone();
        auto detached = result.detach();

        if (result.dim() > 0 && result.size(0) > 0) {
            auto sliced = result[0];
        }

        if (result.numel() > 1) {
            auto flattened = result.flatten();
        }

        if (result.dim() >= 2) {
            auto transposed = result.transpose(0, -1);
        }

        torch::Tensor another_empty = torch::empty_like(result);
        
        if (offset < Size) {
            std::vector<int64_t> new_shape;
            uint8_t new_dims = Data[offset++] % 4;
            for (uint8_t i = 0; i < new_dims && offset < Size; ++i) {
                int64_t new_dim = static_cast<int64_t>(Data[offset++]);
                new_shape.push_back(new_dim);
            }
            if (!new_shape.empty()) {
                torch::Tensor resized_empty = torch::empty(new_shape, options);
            }
        }

        torch::Tensor zero_dim_empty = torch::empty({0}, options);
        torch::Tensor large_empty = torch::empty({1000000}, options);
        torch::Tensor multi_zero = torch::empty({0, 5, 0}, options);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}