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
        
        // Create a tensor to test memory format conversion
        torch::Tensor tensor;
        if (offset < Size) {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default tensor if we've consumed all data
            tensor = torch::ones({2, 3, 4, 5});
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
        }
        
        // Apply memory format to the tensor
        torch::Tensor result = tensor.to(memory_format);
        
        // Test is_contiguous with memory_format
        bool is_contiguous = result.is_contiguous(memory_format);
        
        // Test contiguous with memory_format
        torch::Tensor contiguous_result = result.contiguous(memory_format);
        
        // Test clone with memory_format
        torch::Tensor cloned = tensor.clone(memory_format);
        
        // Test empty_like with memory_format
        torch::Tensor empty_like_tensor = torch::empty_like(tensor, tensor.options().memory_format(memory_format));
        
        // Test zeros_like with memory_format
        torch::Tensor zeros_like_tensor = torch::zeros_like(tensor, tensor.options().memory_format(memory_format));
        
        // Test ones_like with memory_format
        torch::Tensor ones_like_tensor = torch::ones_like(tensor, tensor.options().memory_format(memory_format));
        
        // Test full_like with memory_format
        torch::Scalar fill_value = 3.14;
        torch::Tensor full_like_tensor = torch::full_like(tensor, fill_value, tensor.options().memory_format(memory_format));
        
        // Test rand_like with memory_format
        torch::Tensor rand_like_tensor = torch::rand_like(tensor, tensor.options().memory_format(memory_format));
        
        // Test randn_like with memory_format
        torch::Tensor randn_like_tensor = torch::randn_like(tensor, tensor.options().memory_format(memory_format));
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
