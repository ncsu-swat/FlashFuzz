#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
                required_dims = 4; // Requires 4D tensor (NCHW)
                break;
            case 2:
                memory_format = torch::MemoryFormat::ChannelsLast3d;
                required_dims = 5; // Requires 5D tensor (NCDHW)
                break;
            case 3:
                memory_format = torch::MemoryFormat::Preserve;
                required_dims = 0; // Any dimension works
                break;
            default:
                memory_format = torch::MemoryFormat::Contiguous;
                required_dims = 0;
        }
        
        // Create tensor with appropriate dimensions for the memory format
        torch::Tensor tensor;
        if (required_dims == 4) {
            // Create 4D tensor for ChannelsLast (NCHW format)
            int64_t n = 1 + (Size > offset ? Data[offset++] % 4 : 1);
            int64_t c = 1 + (Size > offset ? Data[offset++] % 8 : 1);
            int64_t h = 1 + (Size > offset ? Data[offset++] % 16 : 1);
            int64_t w = 1 + (Size > offset ? Data[offset++] % 16 : 1);
            tensor = torch::randn({n, c, h, w});
        } else if (required_dims == 5) {
            // Create 5D tensor for ChannelsLast3d (NCDHW format)
            int64_t n = 1 + (Size > offset ? Data[offset++] % 2 : 1);
            int64_t c = 1 + (Size > offset ? Data[offset++] % 4 : 1);
            int64_t d = 1 + (Size > offset ? Data[offset++] % 8 : 1);
            int64_t h = 1 + (Size > offset ? Data[offset++] % 8 : 1);
            int64_t w = 1 + (Size > offset ? Data[offset++] % 8 : 1);
            tensor = torch::randn({n, c, d, h, w});
        } else {
            // Create tensor from fuzzer data for Contiguous/Preserve
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        if (!tensor.defined() || tensor.numel() == 0) {
            return 0;
        }
        
        // Apply memory_format to the tensor using contiguous()
        torch::Tensor result;
        try {
            result = tensor.contiguous(memory_format);
        } catch (...) {
            // Dimension mismatch is expected for some combinations
            return 0;
        }
        
        // Verify the result tensor is valid
        auto sizes = result.sizes();
        auto strides = result.strides();
        auto dtype = result.dtype();
        
        // Perform operations on the result
        if (result.numel() > 0) {
            torch::Tensor doubled = result * 2;
            torch::Tensor summed = result.sum();
        }
        
        // Test to() with memory format
        try {
            torch::Tensor to_result = tensor.to(tensor.options(), false, false, memory_format);
        } catch (...) {
            // May fail for some tensor configurations
        }
        
        // Test clone with memory format
        try {
            torch::Tensor cloned = tensor.clone(memory_format);
            auto cloned_strides = cloned.strides();
        } catch (...) {
            // May fail for incompatible dimensions
        }
        
        // Test is_contiguous with memory format
        bool is_contig = result.is_contiguous(memory_format);
        
        // Test suggest_memory_format
        torch::MemoryFormat suggested = result.suggest_memory_format();
        
        // Test converting between formats if tensor has right dimensions
        if (tensor.dim() == 4) {
            try {
                torch::Tensor channels_last = tensor.contiguous(torch::MemoryFormat::ChannelsLast);
                torch::Tensor back_to_contig = channels_last.contiguous(torch::MemoryFormat::Contiguous);
            } catch (...) {
                // Silently handle failures
            }
        }
        
        if (tensor.dim() == 5) {
            try {
                torch::Tensor channels_last_3d = tensor.contiguous(torch::MemoryFormat::ChannelsLast3d);
                torch::Tensor back_to_contig = channels_last_3d.contiguous(torch::MemoryFormat::Contiguous);
            } catch (...) {
                // Silently handle failures
            }
        }
        
        // Test empty_like with memory format
        try {
            torch::Tensor empty_copy = torch::empty_like(tensor, tensor.options(), memory_format);
        } catch (...) {
            // May fail for some configurations
        }
        
        // Test zeros_like and ones_like with memory format
        try {
            torch::Tensor zeros_copy = torch::zeros_like(tensor, tensor.options(), memory_format);
            torch::Tensor ones_copy = torch::ones_like(tensor, tensor.options(), memory_format);
        } catch (...) {
            // May fail for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}