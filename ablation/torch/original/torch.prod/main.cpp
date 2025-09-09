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

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            torch::prod(input_tensor);
            return 0;
        }

        uint8_t operation_selector = Data[offset++];
        
        if (operation_selector % 2 == 0) {
            torch::prod(input_tensor);
        } else {
            if (offset >= Size) {
                torch::prod(input_tensor);
                return 0;
            }
            
            int64_t dim_raw;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else {
                dim_raw = 0;
                offset = Size;
            }
            
            int dim = static_cast<int>(dim_raw);
            
            bool keepdim = false;
            if (offset < Size) {
                keepdim = (Data[offset++] % 2 == 1);
            }
            
            torch::prod(input_tensor, dim, keepdim);
        }
        
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            if (operation_selector % 2 == 0) {
                torch::prod(input_tensor, dtype);
            } else {
                if (offset >= Size) {
                    torch::prod(input_tensor, dtype);
                    return 0;
                }
                
                int64_t dim_raw;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    dim_raw = 0;
                    offset = Size;
                }
                
                int dim = static_cast<int>(dim_raw);
                
                bool keepdim = false;
                if (offset < Size) {
                    keepdim = (Data[offset++] % 2 == 1);
                }
                
                torch::prod(input_tensor, dim, keepdim, dtype);
            }
        }
        
        auto empty_tensor = torch::empty({0});
        torch::prod(empty_tensor);
        
        auto scalar_tensor = torch::tensor(5.0);
        torch::prod(scalar_tensor);
        
        auto large_tensor = torch::ones({1000});
        torch::prod(large_tensor);
        
        auto negative_tensor = torch::tensor({-1.0, -2.0, -3.0});
        torch::prod(negative_tensor);
        
        auto mixed_tensor = torch::tensor({{1.0, -1.0}, {2.0, 0.0}});
        torch::prod(mixed_tensor, 0);
        torch::prod(mixed_tensor, 1);
        torch::prod(mixed_tensor, -1);
        torch::prod(mixed_tensor, -2);
        
        auto inf_tensor = torch::tensor({std::numeric_limits<float>::infinity(), 1.0});
        torch::prod(inf_tensor);
        
        auto nan_tensor = torch::tensor({std::numeric_limits<float>::quiet_NaN(), 1.0});
        torch::prod(nan_tensor);
        
        auto zero_tensor = torch::zeros({5, 5});
        torch::prod(zero_tensor);
        torch::prod(zero_tensor, 0);
        torch::prod(zero_tensor, 1);
        
        auto bool_tensor = torch::tensor({true, false, true}, torch::kBool);
        torch::prod(bool_tensor);
        
        auto int_tensor = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        torch::prod(int_tensor);
        torch::prod(int_tensor, torch::kFloat64);
        
        auto complex_tensor = torch::tensor({{1.0 + 1.0i, 2.0 - 1.0i}}, torch::kComplexFloat);
        torch::prod(complex_tensor);
        
        auto high_dim_tensor = torch::ones({2, 3, 4, 5});
        for (int d = 0; d < 4; ++d) {
            torch::prod(high_dim_tensor, d);
            torch::prod(high_dim_tensor, d, true);
            torch::prod(high_dim_tensor, d, false);
        }
        
        auto single_element = torch::tensor({42.0});
        torch::prod(single_element, 0);
        torch::prod(single_element, -1);
        
        auto overflow_tensor = torch::full({10}, 1e20f);
        torch::prod(overflow_tensor);
        
        auto underflow_tensor = torch::full({10}, 1e-20f);
        torch::prod(underflow_tensor);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}