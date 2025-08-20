#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply empty_like operation with various options
        if (offset + 1 < Size) {
            // Use the next byte to determine memory format
            uint8_t memory_format_byte = Data[offset++];
            c10::MemoryFormat memory_format = static_cast<c10::MemoryFormat>(memory_format_byte % 4); // 4 memory formats
            
            // Use the next byte to determine pin_memory option if available
            bool pin_memory = false;
            if (offset < Size) {
                pin_memory = Data[offset++] & 0x1;
            }
            
            // Create empty_like tensor with default options
            torch::Tensor result1 = torch::empty_like(input_tensor);
            
            // Create empty_like tensor with specified dtype if we have more data
            if (offset < Size) {
                uint8_t dtype_byte = Data[offset++];
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_byte);
                torch::Tensor result2 = torch::empty_like(input_tensor, 
                                                         torch::TensorOptions()
                                                         .dtype(dtype)
                                                         .memory_format(memory_format)
                                                         .pinned_memory(pin_memory));
            }
            
            // Create empty_like tensor with layout option if we have more data
            if (offset < Size) {
                uint8_t layout_byte = Data[offset++];
                c10::Layout layout = (layout_byte % 2 == 0) ? c10::kStrided : c10::kSparse;
                torch::Tensor result3 = torch::empty_like(input_tensor, 
                                                         torch::TensorOptions()
                                                         .layout(layout)
                                                         .memory_format(memory_format));
            }
            
            // Create empty_like tensor with device option if we have more data
            if (offset < Size) {
                uint8_t device_byte = Data[offset++];
                torch::Device device = (device_byte % 2 == 0) ? torch::kCPU : torch::kCUDA;
                try {
                    torch::Tensor result4 = torch::empty_like(input_tensor, 
                                                             torch::TensorOptions()
                                                             .device(device)
                                                             .memory_format(memory_format));
                } catch (...) {
                    // CUDA might not be available, ignore this error
                }
            }
            
            // Create empty_like tensor with requires_grad option if we have more data
            if (offset < Size) {
                uint8_t requires_grad_byte = Data[offset++];
                bool requires_grad = requires_grad_byte & 0x1;
                torch::Tensor result5 = torch::empty_like(input_tensor, 
                                                         torch::TensorOptions()
                                                         .requires_grad(requires_grad)
                                                         .memory_format(memory_format));
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