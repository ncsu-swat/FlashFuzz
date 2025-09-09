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
            int32_t second_call_raw;
            std::memcpy(&second_call_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            torch::set_num_threads(second_call_raw);
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t third_call_raw;
            std::memcpy(&third_call_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            torch::set_num_threads(third_call_raw);
        }
        
        if (offset < Size) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto result = tensor + 1.0f;
            } catch (...) {
            }
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