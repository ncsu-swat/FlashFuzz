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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create a 4D tensor for conv2d weight
        // Conv2d weight format is [out_channels, in_channels, kernel_height, kernel_width]
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 4D tensor for conv2d weights
        // If not 4D, reshape it to 4D if possible
        if (weight.dim() != 4) {
            // Try to reshape to 4D if we have enough elements
            int64_t total_elements = weight.numel();
            
            // Create a reasonable 4D shape
            int64_t out_channels = std::max(int64_t(1), total_elements / 9);
            int64_t in_channels = std::min(int64_t(3), total_elements / out_channels);
            int64_t kernel_size = std::max(int64_t(1), (int64_t)std::sqrt(total_elements / (out_channels * in_channels)));
            
            // Ensure we don't exceed the total number of elements
            while (out_channels * in_channels * kernel_size * kernel_size > total_elements && out_channels > 1) {
                out_channels--;
            }
            
            // Reshape if possible
            if (out_channels * in_channels * kernel_size * kernel_size <= total_elements) {
                weight = weight.reshape({out_channels, in_channels, kernel_size, kernel_size});
            } else {
                // If reshaping is not possible, create a small valid 4D tensor
                weight = torch::ones({1, 1, 1, 1}, weight.options());
            }
        }
        
        // Get a memory format to convert to
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
                memory_format = torch::MemoryFormat::Preserve;
                break;
            case 3:
                memory_format = torch::MemoryFormat::ChannelsLast3d;
                break;
            default:
                memory_format = torch::MemoryFormat::Contiguous;
        }
        
        // Apply the operation using to() method with memory format
        torch::Tensor result = weight.to(memory_format);
        
        // Perform some operations on the result to ensure it's used
        auto sum = result.sum();
        
        // Try with different memory formats if we have more data
        if (offset + 1 < Size) {
            format_selector = Data[offset++];
            torch::MemoryFormat another_format;
            switch (format_selector % 4) {
                case 0:
                    another_format = torch::MemoryFormat::Contiguous;
                    break;
                case 1:
                    another_format = torch::MemoryFormat::ChannelsLast;
                    break;
                case 2:
                    another_format = torch::MemoryFormat::Preserve;
                    break;
                case 3:
                    another_format = torch::MemoryFormat::ChannelsLast3d;
                    break;
                default:
                    another_format = torch::MemoryFormat::Contiguous;
            }
            
            // Convert again with a different format
            torch::Tensor result2 = result.to(another_format);
            auto sum2 = result2.sum();
        }
        
        // Try with a non-contiguous tensor if we have more data
        if (offset < Size) {
            // Create a non-contiguous tensor by slicing
            if (weight.sizes()[0] > 1 && weight.sizes()[1] > 1) {
                auto non_contiguous_weight = weight.slice(0, 0, weight.sizes()[0], 2).slice(1, 0, weight.sizes()[1], 2);
                torch::Tensor result3 = non_contiguous_weight.to(memory_format);
                auto sum3 = result3.sum();
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
