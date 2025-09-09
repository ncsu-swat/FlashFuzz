#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            torch::flatten(input_tensor);
            return 0;
        }
        
        uint8_t param_byte = Data[offset++];
        uint8_t mode = param_byte & 0x03;
        
        if (mode == 0) {
            torch::flatten(input_tensor);
        }
        else if (mode == 1) {
            if (offset >= Size) {
                torch::flatten(input_tensor, 0);
                return 0;
            }
            int64_t start_dim_raw;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&start_dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else {
                start_dim_raw = Data[offset++];
            }
            
            int64_t start_dim = start_dim_raw;
            torch::flatten(input_tensor, start_dim);
        }
        else if (mode == 2) {
            if (offset + 1 >= Size) {
                torch::flatten(input_tensor, 0, -1);
                return 0;
            }
            
            int64_t start_dim_raw, end_dim_raw;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&start_dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else {
                start_dim_raw = Data[offset++];
            }
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&end_dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else if (offset < Size) {
                end_dim_raw = Data[offset++];
            } else {
                end_dim_raw = -1;
            }
            
            int64_t start_dim = start_dim_raw;
            int64_t end_dim = end_dim_raw;
            torch::flatten(input_tensor, start_dim, end_dim);
        }
        else {
            if (offset >= Size) {
                torch::flatten(input_tensor, 0, -1);
                return 0;
            }
            
            int64_t start_dim_raw, end_dim_raw;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&start_dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else {
                start_dim_raw = Data[offset++];
            }
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&end_dim_raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else if (offset < Size) {
                end_dim_raw = Data[offset++];
            } else {
                end_dim_raw = -1;
            }
            
            int64_t start_dim = start_dim_raw;
            int64_t end_dim = end_dim_raw;
            
            torch::flatten(input_tensor, start_dim, end_dim);
            
            if (input_tensor.dim() > 0) {
                torch::flatten(input_tensor, -input_tensor.dim(), input_tensor.dim() - 1);
            }
            
            torch::flatten(input_tensor, std::numeric_limits<int64_t>::min());
            torch::flatten(input_tensor, std::numeric_limits<int64_t>::max());
            torch::flatten(input_tensor, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}