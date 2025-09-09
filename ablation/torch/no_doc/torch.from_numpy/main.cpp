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

        auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t numpy_flags = Data[offset++];
        
        auto numpy_tensor = tensor.detach().cpu();
        
        if (numpy_flags & 0x01) {
            numpy_tensor = numpy_tensor.contiguous();
        }
        
        if (numpy_flags & 0x02) {
            numpy_tensor = numpy_tensor.transpose(-1, -2);
        }
        
        if (numpy_flags & 0x04 && numpy_tensor.numel() > 0) {
            auto shape = numpy_tensor.sizes().vec();
            if (!shape.empty()) {
                shape[0] = std::max(1L, shape[0] / 2);
                numpy_tensor = numpy_tensor.view(shape);
            }
        }
        
        if (numpy_flags & 0x08) {
            numpy_tensor = numpy_tensor.clone();
        }
        
        auto result_tensor = torch::from_numpy(numpy_tensor);
        
        if (numpy_flags & 0x10) {
            result_tensor = result_tensor.to(torch::kFloat32);
        }
        
        if (numpy_flags & 0x20) {
            result_tensor = result_tensor.sum();
        }
        
        if (numpy_flags & 0x40 && result_tensor.numel() > 1) {
            result_tensor = result_tensor.flatten();
        }
        
        if (numpy_flags & 0x80 && result_tensor.dim() > 0) {
            result_tensor = result_tensor.squeeze();
        }
        
        auto final_result = result_tensor.detach();
        
        if (final_result.numel() > 0) {
            auto sum_val = final_result.sum();
            volatile auto item_val = sum_val.item<double>();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}