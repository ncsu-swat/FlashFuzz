#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset + 1 >= Size) {
            return 0;
        }
        
        uint8_t ctypes_type = Data[offset++];
        ctypes_type = ctypes_type % 5;  // Limit to a few common ctypes types
        
        void* ptr = nullptr;
        
        switch (ctypes_type) {
            case 0:  // c_void_p
                ptr = tensor.data_ptr();
                break;
            case 1:  // c_long
                if (tensor.numel() > 0) {
                    ptr = tensor.data_ptr<int64_t>();
                }
                break;
            case 2:  // c_int
                if (tensor.numel() > 0) {
                    ptr = tensor.data_ptr<int32_t>();
                }
                break;
            case 3:  // c_float
                if (tensor.numel() > 0 && tensor.dtype() == torch::kFloat32) {
                    ptr = tensor.data_ptr<float>();
                }
                break;
            case 4:  // c_double
                if (tensor.numel() > 0 && tensor.dtype() == torch::kFloat64) {
                    ptr = tensor.data_ptr<double>();
                }
                break;
        }
        
        if (ptr) {
            auto size_bytes = tensor.numel() * tensor.element_size();
            auto storage_offset = tensor.storage_offset();
            
            auto tensor_from_ptr = torch::from_blob(
                ptr, 
                tensor.sizes(), 
                tensor.strides(),
                [](void*){},  // No-op deleter since we don't own the memory
                tensor.options()
            );
            
            if (tensor.numel() > 0) {
                auto element = tensor.flatten()[0];
                auto element_from_ptr = tensor_from_ptr.flatten()[0];
                
                if (element.item<float>() != element_from_ptr.item<float>()) {
                    throw std::runtime_error("Data mismatch between original tensor and tensor from ctypes pointer");
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}