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
                    if (min_tensor.numel() == 1) {
                        auto scalar_min = min_tensor.item();
                        auto result = torch::clamp_min(input_tensor, scalar_min);
                    }
                }
            } else {
                auto result = torch::clamp_min(input_tensor, torch::tensor(0.0));
            }
        }
        
        if (offset < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag % 2 == 0) {
                auto input_copy = input_tensor.clone();
                if (offset + sizeof(float) <= Size) {
                    float min_val;
                    std::memcpy(&min_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    input_copy.clamp_min_(min_val);
                } else {
                    input_copy.clamp_min_(1.0f);
                }
            }
        }
        
        if (offset < Size) {
            uint8_t out_flag = Data[offset++];
            if (out_flag % 3 == 0) {
                auto out_tensor = torch::empty_like(input_tensor);
                if (offset + sizeof(double) <= Size) {
                    double min_val;
                    std::memcpy(&min_val, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    torch::clamp_min_out(out_tensor, input_tensor, min_val);
                } else {
                    torch::clamp_min_out(out_tensor, input_tensor, -1.0);
                }
            }
        }
        
        if (input_tensor.numel() > 0 && offset < Size) {
            uint8_t extreme_flag = Data[offset++];
            if (extreme_flag % 4 == 0) {
                auto result1 = torch::clamp_min(input_tensor, std::numeric_limits<double>::lowest());
                auto result2 = torch::clamp_min(input_tensor, std::numeric_limits<double>::max());
                auto result3 = torch::clamp_min(input_tensor, std::numeric_limits<double>::infinity());
                auto result4 = torch::clamp_min(input_tensor, -std::numeric_limits<double>::infinity());
                auto result5 = torch::clamp_min(input_tensor, std::numeric_limits<double>::quiet_NaN());
            }
        }
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            if (offset + sizeof(float) * 2 <= Size) {
                float real_part, imag_part;
                std::memcpy(&real_part, Data + offset, sizeof(float));
                offset += sizeof(float);
                std::memcpy(&imag_part, Data + offset, sizeof(float));
                offset += sizeof(float);
                
                auto complex_min = torch::tensor(std::complex<float>(real_part, imag_part));
                auto result = torch::clamp_min(input_tensor, complex_min);
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