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
            return 0;
        }
        
        int64_t dim0_raw, dim1_raw;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim0_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            dim0_raw = static_cast<int64_t>(Data[offset % Size]);
            offset++;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim1_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            dim1_raw = static_cast<int64_t>(Data[offset % Size]);
            offset++;
        }
        
        int64_t tensor_ndim = input_tensor.dim();
        
        int64_t dim0, dim1;
        
        if (tensor_ndim == 0) {
            dim0 = 0;
            dim1 = 0;
        } else {
            dim0 = dim0_raw % (2 * tensor_ndim) - tensor_ndim;
            dim1 = dim1_raw % (2 * tensor_ndim) - tensor_ndim;
        }
        
        auto result = torch::swapdims(input_tensor, dim0, dim1);
        
        if (tensor_ndim > 0) {
            auto transpose_result = torch::transpose(input_tensor, dim0, dim1);
            
            if (!torch::equal(result, transpose_result)) {
                std::cout << "Mismatch between swapdims and transpose results" << std::endl;
            }
        }
        
        if (tensor_ndim >= 2) {
            auto double_swap = torch::swapdims(result, dim0, dim1);
            
            if (!torch::equal(input_tensor, double_swap)) {
                std::cout << "Double swap did not return original tensor" << std::endl;
            }
        }
        
        if (dim0 == dim1 && !torch::equal(input_tensor, result)) {
            std::cout << "Swapping same dimension should return identical tensor" << std::endl;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}