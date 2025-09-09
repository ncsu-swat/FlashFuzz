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
        
        bool flush_denormal = Data[offset] % 2;
        offset++;
        
        bool original_state = torch::is_flush_denormal();
        
        torch::set_flush_denormal(flush_denormal);
        
        bool new_state = torch::is_flush_denormal();
        
        torch::set_flush_denormal(!flush_denormal);
        torch::set_flush_denormal(flush_denormal);
        
        if (offset < Size) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                torch::set_flush_denormal(true);
                auto result1 = tensor * 1e-40f;
                
                torch::set_flush_denormal(false);
                auto result2 = tensor * 1e-40f;
                
                torch::set_flush_denormal(flush_denormal);
                auto result3 = tensor + torch::tensor(1e-45f);
                
            } catch (const std::exception &e) {
                torch::set_flush_denormal(original_state);
                return 0;
            }
        }
        
        torch::set_flush_denormal(original_state);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}