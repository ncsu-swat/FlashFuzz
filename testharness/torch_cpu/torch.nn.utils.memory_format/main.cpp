#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to select memory format
        if (Size < 1) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Get a byte to select memory format
        uint8_t format_selector = 0;
        if (offset < Size) {
            format_selector = Data[offset++];
        }
        
        // Select memory format based on the byte
        torch::MemoryFormat memory_format;
        switch (format_selector % 4) {
            case 0:
                memory_format = torch::MemoryFormat::Contiguous;
                break;
            case 1:
                memory_format = torch::MemoryFormat::ChannelsLast;
                break;
            case 2:
                memory_format = torch::MemoryFormat::ChannelsLast3d;
                break;
            case 3:
                memory_format = torch::MemoryFormat::Preserve;
                break;
            default:
                memory_format = torch::MemoryFormat::Contiguous;
        }
        
        // Apply memory_format to the tensor
        torch::Tensor result = tensor.to(memory_format);
        
        // Try to access some properties of the result to ensure it's valid
        auto sizes = result.sizes();
        auto strides = result.strides();
        auto dtype = result.dtype();
        
        // Try to perform some operations on the result
        if (result.numel() > 0) {
            torch::Tensor doubled = result * 2;
            torch::Tensor summed = result.sum();
        }
        
        // Try to convert back to contiguous if it's not already
        if (memory_format != torch::MemoryFormat::Contiguous) {
            torch::Tensor contiguous_result = result.contiguous();
        }
        
        // Try to clone with a different memory format
        torch::MemoryFormat another_format = (memory_format == torch::MemoryFormat::Contiguous) ? 
                                            torch::MemoryFormat::ChannelsLast : 
                                            torch::MemoryFormat::Contiguous;
        
        torch::Tensor cloned = result.clone(another_format);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
