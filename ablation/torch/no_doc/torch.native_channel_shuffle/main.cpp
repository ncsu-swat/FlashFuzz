#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        int64_t groups = static_cast<int64_t>(Data[offset] % 16 + 1);
        offset++;
        
        if (offset < Size) {
            int64_t groups_raw;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&groups_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                groups = std::abs(groups_raw) % 1000 + 1;
            }
        }
        
        torch::native_channel_shuffle(input_tensor, groups);
        
        if (input_tensor.dim() >= 1) {
            torch::native_channel_shuffle(input_tensor, 1);
        }
        
        if (input_tensor.dim() >= 2 && input_tensor.size(1) > 0) {
            int64_t channels = input_tensor.size(1);
            for (int64_t g = 1; g <= std::min(channels, static_cast<int64_t>(10)); ++g) {
                if (channels % g == 0) {
                    torch::native_channel_shuffle(input_tensor, g);
                }
            }
        }
        
        if (input_tensor.numel() > 0) {
            torch::native_channel_shuffle(input_tensor, groups);
            torch::native_channel_shuffle(input_tensor, -groups);
            torch::native_channel_shuffle(input_tensor, 0);
            torch::native_channel_shuffle(input_tensor, std::numeric_limits<int64_t>::max());
            torch::native_channel_shuffle(input_tensor, std::numeric_limits<int64_t>::min());
        }
        
        auto empty_tensor = torch::empty({0});
        torch::native_channel_shuffle(empty_tensor, groups);
        
        auto scalar_tensor = torch::tensor(42.0);
        torch::native_channel_shuffle(scalar_tensor, groups);
        
        if (input_tensor.dim() >= 3) {
            auto reshaped = input_tensor.view({-1, input_tensor.size(-1)});
            torch::native_channel_shuffle(reshaped, groups);
        }
        
        if (input_tensor.is_floating_point()) {
            auto nan_tensor = input_tensor.clone();
            nan_tensor.fill_(std::numeric_limits<float>::quiet_NaN());
            torch::native_channel_shuffle(nan_tensor, groups);
            
            auto inf_tensor = input_tensor.clone();
            inf_tensor.fill_(std::numeric_limits<float>::infinity());
            torch::native_channel_shuffle(inf_tensor, groups);
        }
        
        if (input_tensor.dim() >= 1) {
            auto large_groups = input_tensor.size(0) * 2;
            torch::native_channel_shuffle(input_tensor, large_groups);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}