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
        
        uint8_t mode_byte = Data[offset++];
        bool mode = (mode_byte % 2) == 1;
        
        torch::use_deterministic_algorithms(mode);
        
        if (offset < Size) {
            uint8_t warn_byte = Data[offset++];
            bool warn_only = (warn_byte % 2) == 1;
            torch::use_deterministic_algorithms(mode, warn_only);
        }
        
        bool current_mode = torch::are_deterministic_algorithms_enabled();
        
        if (offset < Size) {
            uint8_t toggle_byte = Data[offset++];
            bool new_mode = (toggle_byte % 2) == 1;
            torch::use_deterministic_algorithms(new_mode);
            torch::use_deterministic_algorithms(!new_mode);
        }
        
        torch::use_deterministic_algorithms(true);
        torch::use_deterministic_algorithms(false);
        torch::use_deterministic_algorithms(true, true);
        torch::use_deterministic_algorithms(false, false);
        
        if (offset < Size) {
            for (size_t i = offset; i < Size && i < offset + 10; ++i) {
                bool rapid_mode = (Data[i] % 2) == 1;
                torch::use_deterministic_algorithms(rapid_mode);
            }
        }
        
        torch::use_deterministic_algorithms(false);
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}