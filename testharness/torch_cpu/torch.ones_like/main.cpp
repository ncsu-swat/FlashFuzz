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
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply ones_like operation - basic case
        torch::Tensor result = torch::ones_like(input_tensor);
        
        // Try with different options
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Test with different dtype
            if (option_byte & 0x01) {
                try {
                    torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset % Size]);
                    torch::Tensor result_with_dtype = torch::ones_like(input_tensor, dtype);
                } catch (const std::exception &) {
                    // Expected: some dtype conversions may not be supported
                }
            }
            
            // Test with strided layout (sparse can have issues with ones_like)
            if (option_byte & 0x02) {
                try {
                    torch::Tensor result_with_layout = torch::ones_like(
                        input_tensor, 
                        input_tensor.options().layout(torch::kStrided)
                    );
                } catch (const std::exception &) {
                    // Expected: layout conversion may fail
                }
            }
            
            // Test with different device (CPU only in this context)
            if (option_byte & 0x08) {
                try {
                    torch::Device device = torch::kCPU;
                    torch::Tensor result_with_device = torch::ones_like(
                        input_tensor, 
                        input_tensor.options().device(device)
                    );
                } catch (const std::exception &) {
                    // Expected: device transfer may fail
                }
            }
            
            // Test with requires_grad (only valid for floating point types)
            if (option_byte & 0x10) {
                try {
                    bool requires_grad = (option_byte & 0x20) != 0;
                    // Only set requires_grad if input is floating point
                    if (input_tensor.is_floating_point() || !requires_grad) {
                        torch::Tensor result_with_grad = torch::ones_like(
                            input_tensor, 
                            input_tensor.options().requires_grad(requires_grad)
                        );
                    }
                } catch (const std::exception &) {
                    // Expected: requires_grad may fail for non-floating types
                }
            }
            
            // Test with memory_format
            if (option_byte & 0x40) {
                try {
                    torch::MemoryFormat memory_format;
                    uint8_t format_selector = (option_byte >> 7) & 0x01;
                    
                    switch (format_selector) {
                        case 0:
                            memory_format = torch::MemoryFormat::Contiguous;
                            break;
                        case 1:
                            memory_format = torch::MemoryFormat::Preserve;
                            break;
                        default:
                            memory_format = torch::MemoryFormat::Contiguous;
                    }
                    
                    torch::Tensor result_with_memory_format = torch::ones_like(
                        input_tensor, 
                        input_tensor.options().memory_format(memory_format)
                    );
                } catch (const std::exception &) {
                    // Expected: memory format may not be applicable
                }
            }
        }
        
        // Test with combined options if we have enough data
        if (offset + 2 < Size) {
            uint8_t dtype_byte = Data[offset++];
            uint8_t option_byte = Data[offset++];
            
            try {
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_byte);
                torch::Device device = torch::kCPU;
                
                // Build options carefully
                auto options = input_tensor.options()
                    .dtype(dtype)
                    .layout(torch::kStrided)  // Use strided layout for safety
                    .device(device);
                
                // Only set requires_grad for floating point types
                bool wants_grad = (option_byte & 0x02) != 0;
                if (wants_grad && (dtype == torch::kFloat || dtype == torch::kDouble || 
                                   dtype == torch::kFloat16 || dtype == torch::kBFloat16)) {
                    options = options.requires_grad(true);
                }
                
                torch::Tensor result_with_all_options = torch::ones_like(input_tensor, options);
            } catch (const std::exception &) {
                // Expected: combined options may produce invalid configurations
            }
        }
        
        // Additional test: ChannelsLast memory format for 4D tensors
        if (input_tensor.dim() == 4) {
            try {
                torch::Tensor result_channels_last = torch::ones_like(
                    input_tensor,
                    input_tensor.options().memory_format(torch::MemoryFormat::ChannelsLast)
                );
            } catch (const std::exception &) {
                // Expected: ChannelsLast may not be supported for all dtypes
            }
        }
        
        // Additional test: ChannelsLast3d memory format for 5D tensors
        if (input_tensor.dim() == 5) {
            try {
                torch::Tensor result_channels_last_3d = torch::ones_like(
                    input_tensor,
                    input_tensor.options().memory_format(torch::MemoryFormat::ChannelsLast3d)
                );
            } catch (const std::exception &) {
                // Expected: ChannelsLast3d may not be supported for all dtypes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}