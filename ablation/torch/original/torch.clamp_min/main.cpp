#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t min_type_selector = Data[offset++];
        
        if (min_type_selector % 2 == 0) {
            if (offset + sizeof(double) <= Size) {
                double min_val;
                std::memcpy(&min_val, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                auto result = torch::clamp_min(input_tensor, min_val);
            } else {
                auto result = torch::clamp_min(input_tensor, 0.0);
            }
        } else {
            if (offset < Size) {
                auto min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                try {
                    auto result = torch::clamp_min(input_tensor, min_tensor);
                } catch (const std::exception &e) {
                    auto scalar_min = torch::rand({}).item<double>();
                    auto result = torch::clamp_min(input_tensor, scalar_min);
                }
            } else {
                auto result = torch::clamp_min(input_tensor, 1.0);
            }
        }
        
        if (offset < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag % 3 == 0) {
                auto input_copy = input_tensor.clone();
                if (offset + sizeof(float) <= Size) {
                    float min_val;
                    std::memcpy(&min_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    input_copy.clamp_min_(min_val);
                } else {
                    input_copy.clamp_min_(-1.0f);
                }
            }
        }
        
        if (input_tensor.numel() > 0 && offset < Size) {
            uint8_t extreme_selector = Data[offset++];
            double extreme_vals[] = {
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::lowest(),
                0.0, -0.0, 1e-100, -1e-100, 1e100, -1e100
            };
            
            double extreme_min = extreme_vals[extreme_selector % (sizeof(extreme_vals) / sizeof(extreme_vals[0]))];
            auto result = torch::clamp_min(input_tensor, extreme_min);
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            try {
                auto result = torch::clamp_min(input_tensor, 0.5);
            } catch (const std::exception &e) {
            }
        }
        
        if (input_tensor.numel() == 0) {
            auto result = torch::clamp_min(input_tensor, 42.0);
        }
        
        if (input_tensor.dim() == 0) {
            auto result = torch::clamp_min(input_tensor, -999.999);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}