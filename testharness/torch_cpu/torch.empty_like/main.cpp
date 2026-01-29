#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Use remaining bytes to determine options
        if (offset + 1 >= Size) {
            // Just call basic empty_like
            torch::Tensor result = torch::empty_like(input_tensor);
            return 0;
        }
        
        // Use the next byte to determine memory format
        uint8_t memory_format_byte = Data[offset++];
        c10::MemoryFormat memory_format;
        switch (memory_format_byte % 4) {
            case 0: memory_format = c10::MemoryFormat::Contiguous; break;
            case 1: memory_format = c10::MemoryFormat::Preserve; break;
            case 2: memory_format = c10::MemoryFormat::ChannelsLast; break;
            case 3: memory_format = c10::MemoryFormat::ChannelsLast3d; break;
            default: memory_format = c10::MemoryFormat::Contiguous; break;
        }
        
        // Test 1: Create empty_like tensor with default options
        torch::Tensor result1 = torch::empty_like(input_tensor);
        
        // Test 2: Create empty_like tensor with specified dtype
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_byte);
            
            try {
                // ChannelsLast formats require 4D tensors, ChannelsLast3d requires 5D
                c10::MemoryFormat safe_format = memory_format;
                auto ndim = input_tensor.dim();
                if ((memory_format == c10::MemoryFormat::ChannelsLast && ndim != 4) ||
                    (memory_format == c10::MemoryFormat::ChannelsLast3d && ndim != 5)) {
                    safe_format = c10::MemoryFormat::Contiguous;
                }
                
                torch::Tensor result2 = torch::empty_like(input_tensor, 
                                                         torch::TensorOptions()
                                                         .dtype(dtype)
                                                         .memory_format(safe_format));
            } catch (...) {
                // Some dtype/memory_format combinations may be invalid
            }
        }
        
        // Test 3: Create empty_like tensor with layout option
        if (offset < Size) {
            uint8_t layout_byte = Data[offset++];
            c10::Layout layout = (layout_byte % 2 == 0) ? c10::kStrided : c10::kSparse;
            
            try {
                // Sparse tensors don't support memory_format
                if (layout == c10::kSparse) {
                    torch::Tensor result3 = torch::empty_like(input_tensor, 
                                                             torch::TensorOptions()
                                                             .layout(layout));
                } else {
                    c10::MemoryFormat safe_format = memory_format;
                    auto ndim = input_tensor.dim();
                    if ((memory_format == c10::MemoryFormat::ChannelsLast && ndim != 4) ||
                        (memory_format == c10::MemoryFormat::ChannelsLast3d && ndim != 5)) {
                        safe_format = c10::MemoryFormat::Contiguous;
                    }
                    torch::Tensor result3 = torch::empty_like(input_tensor, 
                                                             torch::TensorOptions()
                                                             .layout(layout)
                                                             .memory_format(safe_format));
                }
            } catch (...) {
                // Some layout conversions may fail
            }
        }
        
        // Test 4: Create empty_like tensor with requires_grad option
        if (offset < Size) {
            uint8_t requires_grad_byte = Data[offset++];
            bool requires_grad = requires_grad_byte & 0x1;
            
            try {
                // requires_grad only works for floating point tensors
                torch::ScalarType dtype = input_tensor.scalar_type();
                if (requires_grad && !at::isFloatingType(dtype)) {
                    dtype = torch::kFloat32;
                }
                
                c10::MemoryFormat safe_format = memory_format;
                auto ndim = input_tensor.dim();
                if ((memory_format == c10::MemoryFormat::ChannelsLast && ndim != 4) ||
                    (memory_format == c10::MemoryFormat::ChannelsLast3d && ndim != 5)) {
                    safe_format = c10::MemoryFormat::Contiguous;
                }
                
                torch::Tensor result5 = torch::empty_like(input_tensor, 
                                                         torch::TensorOptions()
                                                         .dtype(dtype)
                                                         .requires_grad(requires_grad)
                                                         .memory_format(safe_format));
            } catch (...) {
                // Some combinations may be invalid
            }
        }
        
        // Test 5: Combination of multiple options
        if (offset + 1 < Size) {
            uint8_t dtype_byte = Data[offset++];
            uint8_t option_byte = Data[offset++];
            
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_byte);
            bool requires_grad = (option_byte & 0x1) && at::isFloatingType(dtype);
            
            try {
                torch::Tensor result6 = torch::empty_like(input_tensor,
                                                         torch::TensorOptions()
                                                         .dtype(dtype)
                                                         .device(torch::kCPU)
                                                         .layout(c10::kStrided)
                                                         .requires_grad(requires_grad));
            } catch (...) {
                // Some combinations may be invalid
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}