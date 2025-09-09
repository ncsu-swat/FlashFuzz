#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 6) {
            return 0;
        }

        int precision = static_cast<int>(Data[offset++] % 20);
        if (precision < 0) precision = 0;
        
        int threshold = static_cast<int>((Data[offset++] | (Data[offset++] << 8)) % 10000);
        if (threshold < 0) threshold = 0;
        
        int edgeitems = static_cast<int>(Data[offset++] % 50);
        if (edgeitems < 0) edgeitems = 0;
        
        int linewidth = static_cast<int>(Data[offset++] % 200 + 10);
        if (linewidth < 10) linewidth = 10;
        
        uint8_t profile_selector = Data[offset++];
        std::string profile;
        switch (profile_selector % 4) {
            case 0: profile = "default"; break;
            case 1: profile = "short"; break;
            case 2: profile = "full"; break;
            default: profile = ""; break;
        }
        
        if (Size > offset) {
            uint8_t sci_mode_selector = Data[offset++];
            c10::optional<bool> sci_mode;
            switch (sci_mode_selector % 3) {
                case 0: sci_mode = true; break;
                case 1: sci_mode = false; break;
                default: sci_mode = c10::nullopt; break;
            }
            
            if (!profile.empty()) {
                torch::set_printoptions(precision, threshold, edgeitems, linewidth, profile, sci_mode);
            } else {
                torch::set_printoptions(precision, threshold, edgeitems, linewidth, c10::nullopt, sci_mode);
            }
        } else {
            if (!profile.empty()) {
                torch::set_printoptions(precision, threshold, edgeitems, linewidth, profile);
            } else {
                torch::set_printoptions(precision, threshold, edgeitems, linewidth);
            }
        }
        
        if (Size > offset + 10) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                std::ostringstream oss;
                oss << tensor;
                std::string tensor_str = oss.str();
            } catch (...) {
            }
        }
        
        torch::set_printoptions(4, 1000, 3, 80, "default");
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}