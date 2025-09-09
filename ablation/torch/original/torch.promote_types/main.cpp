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
        
        if (offset < Size) {
            uint8_t type3_selector = Data[offset++];
            torch::ScalarType type3 = fuzzer_utils::parseDataType(type3_selector);
            
            torch::ScalarType promoted_type_chain1 = torch::promote_types(torch::promote_types(type1, type2), type3);
            torch::ScalarType promoted_type_chain2 = torch::promote_types(type1, torch::promote_types(type2, type3));
        }
        
        if (offset < Size) {
            torch::ScalarType self_promote = torch::promote_types(type1, type1);
        }
        
        std::vector<torch::ScalarType> all_types = {
            torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16,
            torch::kComplexFloat, torch::kComplexDouble,
            torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64,
            torch::kBool
        };
        
        for (size_t i = 0; i < all_types.size() && offset < Size; ++i) {
            torch::ScalarType mixed_promote = torch::promote_types(type1, all_types[i]);
            if (offset + 1 < Size) {
                torch::ScalarType reverse_promote = torch::promote_types(all_types[i], type1);
                offset++;
            }
        }
        
        if (offset < Size) {
            for (size_t i = 0; i < all_types.size() && i < Size - offset; ++i) {
                for (size_t j = i; j < all_types.size() && j < Size - offset; ++j) {
                    torch::ScalarType cross_promote = torch::promote_types(all_types[i], all_types[j]);
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