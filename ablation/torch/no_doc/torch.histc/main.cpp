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
        
        int64_t bins = 1;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_bins;
            std::memcpy(&raw_bins, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            bins = std::abs(raw_bins) % 1000 + 1;
        }
        
        double min_val = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        double max_val = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        if (std::isnan(min_val) || std::isnan(max_val) || 
            std::isinf(min_val) || std::isinf(max_val)) {
            min_val = 0.0;
            max_val = 1.0;
        }
        
        torch::histc(input_tensor, bins);
        
        torch::histc(input_tensor, bins, min_val);
        
        torch::histc(input_tensor, bins, min_val, max_val);
        
        if (input_tensor.numel() > 0) {
            torch::histc(input_tensor, 1);
            torch::histc(input_tensor, 10000);
        }
        
        if (input_tensor.dtype() == torch::kFloat || 
            input_tensor.dtype() == torch::kDouble ||
            input_val.dtype() == torch::kHalf) {
            
            auto float_tensor = input_tensor.to(torch::kFloat);
            torch::histc(float_tensor, bins, -1e6, 1e6);
            
            if (float_tensor.numel() > 0) {
                auto min_elem = torch::min(float_tensor).item<float>();
                auto max_elem = torch::max(float_tensor).item<float>();
                if (!std::isnan(min_elem) && !std::isnan(max_elem) && 
                    !std::isinf(min_elem) && !std::isinf(max_elem) &&
                    min_elem <= max_elem) {
                    torch::histc(float_tensor, bins, min_elem, max_elem);
                }
            }
        }
        
        auto empty_tensor = torch::empty({0});
        torch::histc(empty_tensor, bins);
        
        auto scalar_tensor = torch::tensor(5.0);
        torch::histc(scalar_tensor, bins);
        
        if (input_tensor.dim() > 0) {
            auto flattened = input_tensor.flatten();
            torch::histc(flattened, bins, min_val, max_val);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}