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
        uint8_t operation_type = operation_selector % 4;

        switch (operation_type) {
            case 0: {
                int num_threads = torch::get_num_interop_threads();
                break;
            }
            case 1: {
                if (offset < Size) {
                    uint8_t thread_count_byte = Data[offset++];
                    int new_thread_count = static_cast<int>(thread_count_byte);
                    torch::set_num_interop_threads(new_thread_count);
                    int retrieved_threads = torch::get_num_interop_threads();
                }
                break;
            }
            case 2: {
                if (offset + sizeof(int32_t) <= Size) {
                    int32_t raw_thread_count;
                    std::memcpy(&raw_thread_count, Data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    torch::set_num_interop_threads(raw_thread_count);
                    int retrieved_threads = torch::get_num_interop_threads();
                }
                break;
            }
            case 3: {
                if (offset + sizeof(int64_t) <= Size) {
                    int64_t large_thread_count;
                    std::memcpy(&large_thread_count, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    int clamped_count = static_cast<int>(large_thread_count);
                    torch::set_num_interop_threads(clamped_count);
                    int retrieved_threads = torch::get_num_interop_threads();
                }
                break;
            }
        }

        if (offset < Size) {
            uint8_t stress_test_selector = Data[offset++];
            if (stress_test_selector % 2 == 0) {
                for (int i = 0; i < 10; ++i) {
                    int threads = torch::get_num_interop_threads();
                }
            }
        }

        if (offset < Size) {
            uint8_t edge_case_selector = Data[offset++];
            uint8_t edge_case_type = edge_case_selector % 6;
            
            switch (edge_case_type) {
                case 0:
                    torch::set_num_interop_threads(-1);
                    torch::get_num_interop_threads();
                    break;
                case 1:
                    torch::set_num_interop_threads(0);
                    torch::get_num_interop_threads();
                    break;
                case 2:
                    torch::set_num_interop_threads(1);
                    torch::get_num_interop_threads();
                    break;
                case 3:
                    torch::set_num_interop_threads(std::numeric_limits<int>::max());
                    torch::get_num_interop_threads();
                    break;
                case 4:
                    torch::set_num_interop_threads(std::numeric_limits<int>::min());
                    torch::get_num_interop_threads();
                    break;
                case 5:
                    torch::set_num_interop_threads(1000000);
                    torch::get_num_interop_threads();
                    break;
            }
        }

        if (offset < Size) {
            uint8_t concurrent_selector = Data[offset++];
            if (concurrent_selector % 3 == 0) {
                int original_threads = torch::get_num_interop_threads();
                torch::set_num_interop_threads(original_threads + 1);
                int new_threads = torch::get_num_interop_threads();
                torch::set_num_interop_threads(original_threads);
                int restored_threads = torch::get_num_interop_threads();
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