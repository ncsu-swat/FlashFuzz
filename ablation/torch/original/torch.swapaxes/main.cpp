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
        
        if (offset + 2 > Size) {
            return 0;
        }
        
        int64_t axis0_raw, axis1_raw;
        std::memcpy(&axis0_raw, Data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        if (offset + sizeof(int64_t) > Size) {
            axis1_raw = axis0_raw + 1;
        } else {
            std::memcpy(&axis1_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        int64_t tensor_ndim = input_tensor.dim();
        
        int64_t axis0 = axis0_raw;
        int64_t axis1 = axis1_raw;
        
        if (tensor_ndim > 0) {
            axis0 = axis0_raw % tensor_ndim;
            axis1 = axis1_raw % tensor_ndim;
            
            if (axis0 < 0) axis0 += tensor_ndim;
            if (axis1 < 0) axis1 += tensor_ndim;
        }
        
        auto result = torch::swapaxes(input_tensor, axis0, axis1);
        
        if (tensor_ndim >= 2) {
            auto transpose_result = torch::transpose(input_tensor, axis0, axis1);
            if (!torch::allclose(result, transpose_result, 1e-5, 1e-8)) {
                std::cout << "swapaxes and transpose results differ" << std::endl;
            }
        }
        
        if (axis0 == axis1) {
            if (!torch::allclose(result, input_tensor, 1e-5, 1e-8)) {
                std::cout << "swapaxes with same axes should return identical tensor" << std::endl;
            }
        }
        
        auto double_swap = torch::swapaxes(result, axis0, axis1);
        if (!torch::allclose(double_swap, input_tensor, 1e-5, 1e-8)) {
            std::cout << "double swapaxes should return original tensor" << std::endl;
        }
        
        if (tensor_ndim > 0) {
            for (int64_t i = 0; i < tensor_ndim; ++i) {
                if (i != axis0 && i != axis1) {
                    if (result.size(i) != input_tensor.size(i)) {
                        std::cout << "non-swapped dimension size changed" << std::endl;
                    }
                }
            }
            
            if (result.size(axis0) != input_tensor.size(axis1) || 
                result.size(axis1) != input_tensor.size(axis0)) {
                std::cout << "swapped dimensions have incorrect sizes" << std::endl;
            }
        }
        
        torch::swapaxes(input_tensor, -tensor_ndim, tensor_ndim - 1);
        torch::swapaxes(input_tensor, tensor_ndim - 1, -tensor_ndim);
        
        torch::swapaxes(input_tensor, axis0_raw, axis1_raw);
        
        if (tensor_ndim == 0) {
            torch::swapaxes(input_tensor, 0, 0);
            torch::swapaxes(input_tensor, -1, 1);
        }
        
        if (tensor_ndim == 1) {
            torch::swapaxes(input_tensor, 0, 0);
            torch::swapaxes(input_tensor, -1, -1);
            torch::swapaxes(input_tensor, 0, -1);
        }
        
        torch::swapaxes(input_tensor, 1000000, -1000000);
        torch::swapaxes(input_tensor, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min());
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}