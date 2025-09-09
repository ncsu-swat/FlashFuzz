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

        uint8_t operation_selector = Data[offset++];
        
        int num_threads = torch::get_num_interop_threads();
        
        if (Size >= 2) {
            uint8_t set_threads_selector = Data[offset++];
            int new_thread_count = static_cast<int>(set_threads_selector);
            
            if (new_thread_count == 0) {
                new_thread_count = 1;
            }
            
            torch::set_num_interop_threads(new_thread_count);
            
            int updated_threads = torch::get_num_interop_threads();
            
            torch::set_num_interop_threads(num_threads);
        }
        
        if (Size >= 3) {
            uint8_t negative_selector = Data[offset++];
            int negative_value = -static_cast<int>(negative_selector);
            
            torch::set_num_interop_threads(negative_value);
            int result_negative = torch::get_num_interop_threads();
            torch::set_num_interop_threads(num_threads);
        }
        
        if (Size >= 7) {
            uint32_t large_value;
            if (offset + sizeof(uint32_t) <= Size) {
                std::memcpy(&large_value, Data + offset, sizeof(uint32_t));
                offset += sizeof(uint32_t);
                
                torch::set_num_interop_threads(static_cast<int>(large_value));
                int result_large = torch::get_num_interop_threads();
                torch::set_num_interop_threads(num_threads);
            }
        }
        
        if (Size >= 11) {
            int64_t very_large_value;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&very_large_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                torch::set_num_interop_threads(static_cast<int>(very_large_value));
                int result_very_large = torch::get_num_interop_threads();
                torch::set_num_interop_threads(num_threads);
            }
        }
        
        torch::set_num_interop_threads(0);
        int zero_result = torch::get_num_interop_threads();
        torch::set_num_interop_threads(num_threads);
        
        torch::set_num_interop_threads(-1);
        int neg_one_result = torch::get_num_interop_threads();
        torch::set_num_interop_threads(num_threads);
        
        torch::set_num_interop_threads(std::numeric_limits<int>::max());
        int max_int_result = torch::get_num_interop_threads();
        torch::set_num_interop_threads(num_threads);
        
        torch::set_num_interop_threads(std::numeric_limits<int>::min());
        int min_int_result = torch::get_num_interop_threads();
        torch::set_num_interop_threads(num_threads);
        
        for (int i = 0; i < 10; ++i) {
            int current = torch::get_num_interop_threads();
        }
        
        if (Size >= offset + 1) {
            uint8_t thread_pattern = Data[offset++];
            for (int i = 0; i < 5; ++i) {
                int pattern_value = (thread_pattern + i) % 256;
                torch::set_num_interop_threads(pattern_value);
                int pattern_result = torch::get_num_interop_threads();
            }
            torch::set_num_interop_threads(num_threads);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}