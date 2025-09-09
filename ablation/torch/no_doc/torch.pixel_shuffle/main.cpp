#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t upscale_factor_byte = Data[offset++];
        int64_t upscale_factor = static_cast<int64_t>(upscale_factor_byte % 16) + 1;
        
        if (upscale_factor <= 0) {
            upscale_factor = 1;
        }
        
        if (input_tensor.dim() < 3) {
            std::vector<int64_t> new_shape;
            if (input_tensor.dim() == 0) {
                new_shape = {1, upscale_factor * upscale_factor, 1, 1};
            } else if (input_tensor.dim() == 1) {
                new_shape = {1, upscale_factor * upscale_factor, input_tensor.size(0), 1};
            } else if (input_tensor.dim() == 2) {
                new_shape = {1, upscale_factor * upscale_factor, input_tensor.size(0), input_tensor.size(1)};
            }
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        if (input_tensor.dim() == 3) {
            std::vector<int64_t> new_shape = {1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)};
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        if (input_tensor.dim() > 4) {
            std::vector<int64_t> new_shape;
            int64_t batch_size = 1;
            for (int i = 0; i < input_tensor.dim() - 3; ++i) {
                batch_size *= input_tensor.size(i);
            }
            new_shape.push_back(batch_size);
            for (int i = input_tensor.dim() - 3; i < input_tensor.dim(); ++i) {
                new_shape.push_back(input_tensor.size(i));
            }
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        if (input_tensor.dim() == 4 && input_tensor.size(1) % (upscale_factor * upscale_factor) != 0) {
            int64_t channels = input_tensor.size(1);
            int64_t target_channels = ((channels / (upscale_factor * upscale_factor)) + 1) * (upscale_factor * upscale_factor);
            if (target_channels <= 0) {
                target_channels = upscale_factor * upscale_factor;
            }
            std::vector<int64_t> new_shape = {input_tensor.size(0), target_channels, input_tensor.size(2), input_tensor.size(3)};
            auto expanded_tensor = torch::zeros(new_shape, input_tensor.options());
            int64_t copy_channels = std::min(channels, target_channels);
            expanded_tensor.narrow(1, 0, copy_channels).copy_(input_tensor.narrow(1, 0, copy_channels));
            input_tensor = expanded_tensor;
        }
        
        auto result = torch::pixel_shuffle(input_tensor, upscale_factor);
        
        if (offset < Size) {
            int64_t negative_upscale = -static_cast<int64_t>(Data[offset % Size] % 10 + 1);
            try {
                auto negative_result = torch::pixel_shuffle(input_tensor, negative_upscale);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            int64_t zero_upscale = 0;
            try {
                auto zero_result = torch::pixel_shuffle(input_tensor, zero_upscale);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            int64_t large_upscale = static_cast<int64_t>(Data[offset % Size]) * 100 + 1000;
            try {
                auto large_result = torch::pixel_shuffle(input_tensor, large_upscale);
            } catch (...) {
            }
        }
        
        auto empty_tensor = torch::empty({0, 0, 0, 0}, input_tensor.options());
        try {
            auto empty_result = torch::pixel_shuffle(empty_tensor, upscale_factor);
        } catch (...) {
        }
        
        auto single_element = torch::ones({1, upscale_factor * upscale_factor, 1, 1}, input_tensor.options());
        try {
            auto single_result = torch::pixel_shuffle(single_element, upscale_factor);
        } catch (...) {
        }
        
        if (input_tensor.numel() > 0) {
            auto mismatched_channels = torch::ones({1, upscale_factor * upscale_factor - 1, 2, 2}, input_tensor.options());
            try {
                auto mismatch_result = torch::pixel_shuffle(mismatched_channels, upscale_factor);
            } catch (...) {
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}