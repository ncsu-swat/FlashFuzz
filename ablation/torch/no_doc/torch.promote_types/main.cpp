#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        uint8_t type1_selector = Data[offset++];
        uint8_t type2_selector = Data[offset++];
        
        torch::ScalarType type1 = fuzzer_utils::parseDataType(type1_selector);
        torch::ScalarType type2 = fuzzer_utils::parseDataType(type2_selector);
        
        torch::ScalarType promoted_type = torch::promote_types(type1, type2);
        
        if (Size >= 4) {
            uint8_t type3_selector = Data[offset++];
            uint8_t type4_selector = Data[offset++];
            
            torch::ScalarType type3 = fuzzer_utils::parseDataType(type3_selector);
            torch::ScalarType type4 = fuzzer_utils::parseDataType(type4_selector);
            
            torch::ScalarType promoted_type2 = torch::promote_types(type3, type4);
            torch::ScalarType double_promoted = torch::promote_types(promoted_type, promoted_type2);
        }
        
        torch::ScalarType self_promoted = torch::promote_types(type1, type1);
        
        if (Size >= 6) {
            for (size_t i = 0; i < std::min(Size - offset, static_cast<size_t>(10)); ++i) {
                if (offset + i < Size) {
                    torch::ScalarType random_type = fuzzer_utils::parseDataType(Data[offset + i]);
                    torch::ScalarType chain_promoted = torch::promote_types(promoted_type, random_type);
                }
            }
        }
        
        std::vector<torch::ScalarType> all_types = {
            torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16,
            torch::kComplexFloat, torch::kComplexDouble,
            torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64,
            torch::kBool
        };
        
        for (const auto& test_type : all_types) {
            torch::ScalarType result1 = torch::promote_types(type1, test_type);
            torch::ScalarType result2 = torch::promote_types(test_type, type1);
        }
        
        if (Size >= 8) {
            size_t remaining = Size - offset;
            for (size_t i = 0; i < remaining - 1; i += 2) {
                if (offset + i + 1 < Size) {
                    torch::ScalarType t1 = fuzzer_utils::parseDataType(Data[offset + i]);
                    torch::ScalarType t2 = fuzzer_utils::parseDataType(Data[offset + i + 1]);
                    torch::ScalarType promoted = torch::promote_types(t1, t2);
                    torch::ScalarType reverse_promoted = torch::promote_types(t2, t1);
                }
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