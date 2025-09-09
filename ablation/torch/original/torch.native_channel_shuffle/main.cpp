#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 5) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t groups_byte = Data[offset++];
        int64_t groups = static_cast<int64_t>(groups_byte);
        if (groups == 0) {
            groups = 1;
        }
        
        if (groups < 0) {
            groups = -groups;
        }
        
        if (groups > 1000) {
            groups = groups % 1000 + 1;
        }
        
        auto result = torch::native_channel_shuffle(input_tensor, groups);
        
        if (offset < Size) {
            uint8_t negative_groups_byte = Data[offset++];
            int64_t negative_groups = -static_cast<int64_t>(negative_groups_byte);
            if (negative_groups == 0) {
                negative_groups = -1;
            }
            torch::native_channel_shuffle(input_tensor, negative_groups);
        }
        
        if (offset < Size) {
            uint8_t large_groups_byte = Data[offset++];
            int64_t large_groups = static_cast<int64_t>(large_groups_byte) * 10000;
            torch::native_channel_shuffle(input_tensor, large_groups);
        }
        
        if (offset < Size) {
            uint8_t zero_groups_byte = Data[offset++];
            torch::native_channel_shuffle(input_tensor, 0);
        }
        
        if (input_tensor.dim() >= 2) {
            int64_t channels = input_tensor.size(1);
            if (channels > 0) {
                torch::native_channel_shuffle(input_tensor, channels);
                torch::native_channel_shuffle(input_tensor, channels + 1);
                torch::native_channel_shuffle(input_tensor, channels * 2);
            }
        }
        
        torch::native_channel_shuffle(input_tensor, 1);
        torch::native_channel_shuffle(input_tensor, 2);
        torch::native_channel_shuffle(input_tensor, 3);
        torch::native_channel_shuffle(input_tensor, 4);
        
        if (offset < Size) {
            auto empty_tensor = torch::empty({0, 0, 0, 0});
            uint8_t empty_groups_byte = Data[offset++];
            int64_t empty_groups = static_cast<int64_t>(empty_groups_byte) + 1;
            torch::native_channel_shuffle(empty_tensor, empty_groups);
        }
        
        if (offset < Size) {
            auto scalar_tensor = torch::scalar_tensor(1.0);
            uint8_t scalar_groups_byte = Data[offset++];
            int64_t scalar_groups = static_cast<int64_t>(scalar_groups_byte) + 1;
            torch::native_channel_shuffle(scalar_tensor, scalar_groups);
        }
        
        if (offset < Size) {
            auto one_d_tensor = torch::randn({10});
            uint8_t one_d_groups_byte = Data[offset++];
            int64_t one_d_groups = static_cast<int64_t>(one_d_groups_byte) + 1;
            torch::native_channel_shuffle(one_d_tensor, one_d_groups);
        }
        
        if (offset < Size) {
            auto large_tensor = torch::randn({1, 1000, 1, 1});
            uint8_t large_groups_byte = Data[offset++];
            int64_t large_tensor_groups = static_cast<int64_t>(large_groups_byte) + 1;
            torch::native_channel_shuffle(large_tensor, large_tensor_groups);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}