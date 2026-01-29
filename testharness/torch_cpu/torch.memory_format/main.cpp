#include "fuzzer_utils.h"
#include <iostream>

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
        // Need at least 2 bytes for format selection and tensor creation
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Get a byte to select memory format
        uint8_t format_selector = Data[offset++];
        
        // Select memory format based on the byte
        torch::MemoryFormat memory_format;
        int required_dims;
        
        switch (format_selector % 4) {
            case 0:
                memory_format = torch::MemoryFormat::Contiguous;
                required_dims = 0; // Any dimension works
                break;
            case 1:
                memory_format = torch::MemoryFormat::ChannelsLast;
                required_dims = 4; // Requires 4D tensor
                break;
            case 2:
                memory_format = torch::MemoryFormat::ChannelsLast3d;
                required_dims = 5; // Requires 5D tensor
                break;
            case 3:
            default:
                memory_format = torch::MemoryFormat::Preserve;
                required_dims = 0; // Any dimension works
                break;
        }
        
        // Create tensor with appropriate dimensions for the memory format
        torch::Tensor tensor;
        if (required_dims == 4) {
            // Create 4D tensor for ChannelsLast
            int64_t n = (offset < Size) ? (Data[offset++] % 4) + 1 : 2;
            int64_t c = (offset < Size) ? (Data[offset++] % 8) + 1 : 3;
            int64_t h = (offset < Size) ? (Data[offset++] % 8) + 1 : 4;
            int64_t w = (offset < Size) ? (Data[offset++] % 8) + 1 : 5;
            tensor = torch::randn({n, c, h, w});
        } else if (required_dims == 5) {
            // Create 5D tensor for ChannelsLast3d
            int64_t n = (offset < Size) ? (Data[offset++] % 4) + 1 : 2;
            int64_t c = (offset < Size) ? (Data[offset++] % 8) + 1 : 3;
            int64_t d = (offset < Size) ? (Data[offset++] % 4) + 1 : 2;
            int64_t h = (offset < Size) ? (Data[offset++] % 4) + 1 : 3;
            int64_t w = (offset < Size) ? (Data[offset++] % 4) + 1 : 4;
            tensor = torch::randn({n, c, d, h, w});
        } else {
            // Use fuzzer-generated tensor for Contiguous/Preserve
            tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            if (tensor.numel() == 0) {
                tensor = torch::randn({2, 3, 4});
            }
        }
        
        // Apply memory format to the tensor using to()
        try {
            torch::Tensor result = tensor.to(memory_format);
            
            // Test is_contiguous with memory_format
            bool is_contiguous = result.is_contiguous(memory_format);
            (void)is_contiguous;
            
            // Test contiguous with memory_format
            torch::Tensor contiguous_result = result.contiguous(memory_format);
        } catch (const c10::Error&) {
            // Silently handle dimension mismatch errors
        }
        
        // Test clone with memory_format
        try {
            torch::Tensor cloned = tensor.clone(memory_format);
        } catch (const c10::Error&) {
            // Silently handle errors
        }
        
        // Test empty_like with memory_format
        try {
            torch::Tensor empty_like_tensor = torch::empty_like(tensor, tensor.options().memory_format(memory_format));
        } catch (const c10::Error&) {
            // Silently handle errors
        }
        
        // Test zeros_like with memory_format
        try {
            torch::Tensor zeros_like_tensor = torch::zeros_like(tensor, tensor.options().memory_format(memory_format));
        } catch (const c10::Error&) {
            // Silently handle errors
        }
        
        // Test ones_like with memory_format
        try {
            torch::Tensor ones_like_tensor = torch::ones_like(tensor, tensor.options().memory_format(memory_format));
        } catch (const c10::Error&) {
            // Silently handle errors
        }
        
        // Test full_like with memory_format
        try {
            torch::Scalar fill_value = 3.14;
            torch::Tensor full_like_tensor = torch::full_like(tensor, fill_value, tensor.options().memory_format(memory_format));
        } catch (const c10::Error&) {
            // Silently handle errors
        }
        
        // Test rand_like with memory_format (only for float types)
        try {
            if (tensor.is_floating_point()) {
                torch::Tensor rand_like_tensor = torch::rand_like(tensor, tensor.options().memory_format(memory_format));
            }
        } catch (const c10::Error&) {
            // Silently handle errors
        }
        
        // Test randn_like with memory_format (only for float types)
        try {
            if (tensor.is_floating_point()) {
                torch::Tensor randn_like_tensor = torch::randn_like(tensor, tensor.options().memory_format(memory_format));
            }
        } catch (const c10::Error&) {
            // Silently handle errors
        }
        
        // Test suggest_memory_format
        torch::MemoryFormat suggested = tensor.suggest_memory_format();
        (void)suggested;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}