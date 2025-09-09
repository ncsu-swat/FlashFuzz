#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        int32_t num_threads_raw;
        std::memcpy(&num_threads_raw, Data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        
        torch::set_num_threads(num_threads_raw);
        
        if (offset < Size) {
            int32_t second_value;
            std::memcpy(&second_value, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            torch::set_num_threads(second_value);
        }
        
        torch::set_num_threads(0);
        torch::set_num_threads(-1);
        torch::set_num_threads(1);
        torch::set_num_threads(std::numeric_limits<int32_t>::max());
        torch::set_num_threads(std::numeric_limits<int32_t>::min());
        
        if (offset + 1 < Size) {
            uint8_t thread_count = Data[offset];
            torch::set_num_threads(static_cast<int>(thread_count));
            offset++;
        }
        
        if (offset + 2 < Size) {
            uint16_t thread_count_16;
            std::memcpy(&thread_count_16, Data + offset, sizeof(uint16_t));
            torch::set_num_threads(static_cast<int>(thread_count_16));
            offset += sizeof(uint16_t);
        }
        
        torch::set_num_threads(1);
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}