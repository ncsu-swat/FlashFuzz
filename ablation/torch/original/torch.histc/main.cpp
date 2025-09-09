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
        
        int bins = 1;
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&bins, Data + offset, sizeof(int));
            offset += sizeof(int);
            bins = std::abs(bins) % 1000 + 1;
        }
        
        double min_val = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        double max_val = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (min_val > max_val && max_val != 0.0) {
            std::swap(min_val, max_val);
        }
        
        torch::histc(input_tensor, bins);
        
        torch::histc(input_tensor, bins, min_val);
        
        torch::histc(input_tensor, bins, min_val, max_val);
        
        if (offset < Size) {
            uint8_t test_case = Data[offset] % 8;
            
            switch (test_case) {
                case 0:
                    torch::histc(input_tensor, 1, -1e10, 1e10);
                    break;
                case 1:
                    torch::histc(input_tensor, 10000, min_val, max_val);
                    break;
                case 2:
                    torch::histc(input_tensor, bins, std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());
                    break;
                case 3:
                    torch::histc(input_tensor, bins, std::numeric_limits<double>::quiet_NaN(), max_val);
                    break;
                case 4:
                    torch::histc(input_tensor, bins, min_val, std::numeric_limits<double>::quiet_NaN());
                    break;
                case 5:
                    torch::histc(input_tensor, bins, std::numeric_limits<double>::infinity(), max_val);
                    break;
                case 6:
                    torch::histc(input_tensor, bins, min_val, std::numeric_limits<double>::infinity());
                    break;
                case 7:
                    torch::histc(input_tensor, bins, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
                    break;
            }
        }
        
        if (input_tensor.numel() > 0) {
            auto flattened = input_tensor.flatten();
            torch::histc(flattened, bins, min_val, max_val);
        }
        
        if (input_tensor.dtype() != torch::kBool && input_tensor.dtype() != torch::kComplexFloat && input_tensor.dtype() != torch::kComplexDouble) {
            auto converted = input_tensor.to(torch::kFloat);
            torch::histc(converted, bins, min_val, max_val);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}