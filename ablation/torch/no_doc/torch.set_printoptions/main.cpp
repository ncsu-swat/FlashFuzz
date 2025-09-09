#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 8) {
            return 0;
        }
        
        int precision = static_cast<int>(Data[offset++] % 20);
        int threshold = static_cast<int>(Data[offset++] % 2000);
        int edgeitems = static_cast<int>(Data[offset++] % 10);
        int linewidth = static_cast<int>(Data[offset++] % 200 + 1);
        bool profile_default = (Data[offset++] % 2) == 1;
        bool sci_mode_default = (Data[offset++] % 2) == 1;
        
        uint8_t profile_selector = Data[offset++];
        std::string profile;
        switch (profile_selector % 3) {
            case 0: profile = "default"; break;
            case 1: profile = "short"; break;
            case 2: profile = "full"; break;
        }
        
        uint8_t sci_mode_selector = Data[offset++];
        c10::optional<bool> sci_mode;
        switch (sci_mode_selector % 3) {
            case 0: sci_mode = c10::nullopt; break;
            case 1: sci_mode = true; break;
            case 2: sci_mode = false; break;
        }
        
        torch::set_printoptions(precision, threshold, edgeitems, linewidth, profile, sci_mode);
        
        if (offset < Size) {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            std::ostringstream oss;
            oss << tensor;
        }
        
        torch::set_printoptions(-1, -1000, -5, 0, "invalid", c10::nullopt);
        
        torch::set_printoptions(1000000, 1000000, 1000000, 1000000, "default", true);
        
        torch::set_printoptions(0, 0, 0, 1, "short", false);
        
        if (offset + 4 <= Size) {
            int32_t extreme_precision;
            std::memcpy(&extreme_precision, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            torch::set_printoptions(extreme_precision, threshold, edgeitems, linewidth, profile, sci_mode);
        }
        
        if (offset + 4 <= Size) {
            int32_t extreme_threshold;
            std::memcpy(&extreme_threshold, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            torch::set_printoptions(precision, extreme_threshold, edgeitems, linewidth, profile, sci_mode);
        }
        
        if (offset + 4 <= Size) {
            int32_t extreme_edgeitems;
            std::memcpy(&extreme_edgeitems, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            torch::set_printoptions(precision, threshold, extreme_edgeitems, linewidth, profile, sci_mode);
        }
        
        if (offset + 4 <= Size) {
            int32_t extreme_linewidth;
            std::memcpy(&extreme_linewidth, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            torch::set_printoptions(precision, threshold, edgeitems, extreme_linewidth, profile, sci_mode);
        }
        
        std::vector<std::string> invalid_profiles = {"", "invalid", "unknown", "test", "bad_profile"};
        for (const auto& invalid_profile : invalid_profiles) {
            torch::set_printoptions(precision, threshold, edgeitems, linewidth, invalid_profile, sci_mode);
        }
        
        torch::set_printoptions();
        
        if (offset < Size) {
            try {
                auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                std::ostringstream oss2;
                oss2 << tensor2;
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