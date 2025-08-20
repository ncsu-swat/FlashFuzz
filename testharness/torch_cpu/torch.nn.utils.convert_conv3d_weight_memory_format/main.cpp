#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a weight tensor for Conv3d
        // Conv3d weight tensor should be 5D: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract memory format options from the remaining data
        torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous;
        if (offset < Size) {
            uint8_t format_selector = Data[offset++];
            // Choose between different memory formats
            switch (format_selector % 3) {
                case 0:
                    memory_format = torch::MemoryFormat::Contiguous;
                    break;
                case 1:
                    memory_format = torch::MemoryFormat::ChannelsLast3d;
                    break;
                case 2:
                    memory_format = torch::MemoryFormat::Preserve;
                    break;
            }
        }
        
        // Try to convert the weight tensor to the specified memory format
        torch::Tensor converted_weight;
        
        // Call the convert_conv3d_weight_memory_format function
        converted_weight = torch::convert_conv3d_weight_memory_format(weight, memory_format);
        
        // Verify the conversion worked by checking the memory format
        if (memory_format != torch::MemoryFormat::Preserve) {
            bool is_correct_format = (converted_weight.suggest_memory_format() == memory_format);
            
            // Use the converted tensor to ensure it's not optimized away
            if (converted_weight.defined()) {
                auto sum = converted_weight.sum();
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