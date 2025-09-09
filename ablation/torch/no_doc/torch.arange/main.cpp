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

        uint8_t variant = Data[offset++] % 6;
        
        if (variant == 0) {
            if (offset + 8 > Size) return 0;
            int64_t end_raw;
            std::memcpy(&end_raw, Data + offset, 8);
            offset += 8;
            
            double end = static_cast<double>(end_raw) / 1000.0;
            
            torch::arange(end);
        }
        else if (variant == 1) {
            if (offset + 16 > Size) return 0;
            int64_t start_raw, end_raw;
            std::memcpy(&start_raw, Data + offset, 8);
            offset += 8;
            std::memcpy(&end_raw, Data + offset, 8);
            offset += 8;
            
            double start = static_cast<double>(start_raw) / 1000.0;
            double end = static_cast<double>(end_raw) / 1000.0;
            
            torch::arange(start, end);
        }
        else if (variant == 2) {
            if (offset + 24 > Size) return 0;
            int64_t start_raw, end_raw, step_raw;
            std::memcpy(&start_raw, Data + offset, 8);
            offset += 8;
            std::memcpy(&end_raw, Data + offset, 8);
            offset += 8;
            std::memcpy(&step_raw, Data + offset, 8);
            offset += 8;
            
            double start = static_cast<double>(start_raw) / 1000.0;
            double end = static_cast<double>(end_raw) / 1000.0;
            double step = static_cast<double>(step_raw) / 1000.0;
            
            torch::arange(start, end, step);
        }
        else if (variant == 3) {
            if (offset + 9 > Size) return 0;
            int64_t end_raw;
            std::memcpy(&end_raw, Data + offset, 8);
            offset += 8;
            
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            
            double end = static_cast<double>(end_raw) / 1000.0;
            
            auto options = torch::TensorOptions().dtype(dtype);
            torch::arange(end, options);
        }
        else if (variant == 4) {
            if (offset + 17 > Size) return 0;
            int64_t start_raw, end_raw;
            std::memcpy(&start_raw, Data + offset, 8);
            offset += 8;
            std::memcpy(&end_raw, Data + offset, 8);
            offset += 8;
            
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            
            double start = static_cast<double>(start_raw) / 1000.0;
            double end = static_cast<double>(end_raw) / 1000.0;
            
            auto options = torch::TensorOptions().dtype(dtype);
            torch::arange(start, end, options);
        }
        else if (variant == 5) {
            if (offset + 25 > Size) return 0;
            int64_t start_raw, end_raw, step_raw;
            std::memcpy(&start_raw, Data + offset, 8);
            offset += 8;
            std::memcpy(&end_raw, Data + offset, 8);
            offset += 8;
            std::memcpy(&step_raw, Data + offset, 8);
            offset += 8;
            
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            
            double start = static_cast<double>(start_raw) / 1000.0;
            double end = static_cast<double>(end_raw) / 1000.0;
            double step = static_cast<double>(step_raw) / 1000.0;
            
            auto options = torch::TensorOptions().dtype(dtype);
            torch::arange(start, end, step, options);
        }

        if (offset < Size) {
            int64_t int_end_raw;
            std::memcpy(&int_end_raw, Data + offset, std::min(size_t(8), Size - offset));
            int64_t int_end = int_end_raw % 10000;
            torch::arange(int_end);
        }

        if (offset + 8 < Size) {
            uint32_t special_case = Data[offset] % 10;
            offset++;
            
            if (special_case == 0) {
                torch::arange(0);
            }
            else if (special_case == 1) {
                torch::arange(-1000, 1000);
            }
            else if (special_case == 2) {
                torch::arange(1000000);
            }
            else if (special_case == 3) {
                torch::arange(0.0, 1.0, 0.0001);
            }
            else if (special_case == 4) {
                torch::arange(-1e10, 1e10, 1e8);
            }
            else if (special_case == 5) {
                torch::arange(std::numeric_limits<double>::lowest() / 1e10, 
                             std::numeric_limits<double>::max() / 1e10, 
                             std::numeric_limits<double>::max() / 1e12);
            }
            else if (special_case == 6) {
                torch::arange(1.0, 0.0, 1.0);
            }
            else if (special_case == 7) {
                torch::arange(0.0, 1.0, -1.0);
            }
            else if (special_case == 8) {
                torch::arange(1.0, 10.0, 0.0);
            }
            else if (special_case == 9) {
                torch::arange(std::numeric_limits<double>::infinity());
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