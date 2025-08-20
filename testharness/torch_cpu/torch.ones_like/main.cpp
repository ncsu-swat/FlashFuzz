#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply ones_like operation
        torch::Tensor result = torch::ones_like(input_tensor);
        
        // Try with different options
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Test with different dtype
            if (option_byte & 0x01) {
                torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset % Size]);
                torch::Tensor result_with_dtype = torch::ones_like(input_tensor, dtype);
            }
            
            // Test with different layout
            if (option_byte & 0x02) {
                torch::Layout layout = (option_byte & 0x04) ? torch::kSparse : torch::kStrided;
                torch::Tensor result_with_layout = torch::ones_like(input_tensor, input_tensor.options().layout(layout));
            }
            
            // Test with different device
            if (option_byte & 0x08) {
                torch::Device device = torch::kCPU;
                torch::Tensor result_with_device = torch::ones_like(input_tensor, input_tensor.options().device(device));
            }
            
            // Test with different requires_grad
            if (option_byte & 0x10) {
                bool requires_grad = (option_byte & 0x20) != 0;
                torch::Tensor result_with_grad = torch::ones_like(input_tensor, input_tensor.options().requires_grad(requires_grad));
            }
            
            // Test with memory_format
            if (option_byte & 0x40) {
                torch::MemoryFormat memory_format;
                uint8_t format_selector = (option_byte & 0x80) ? 1 : 0;
                
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
            }
        }
        
        // Test with all options combined if we have enough data
        if (offset + 2 < Size) {
            uint8_t dtype_byte = Data[offset++];
            uint8_t option_byte = Data[offset++];
            
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_byte);
            torch::Layout layout = (option_byte & 0x01) ? torch::kSparse : torch::kStrided;
            torch::Device device = torch::kCPU;
            bool requires_grad = (option_byte & 0x02) != 0;
            
            torch::Tensor result_with_all_options = torch::ones_like(
                input_tensor,
                input_tensor.options()
                    .dtype(dtype)
                    .layout(layout)
                    .device(device)
                    .requires_grad(requires_grad)
            );
        }
        
        // Verify that all elements in the result tensor are ones
        if (result.numel() > 0) {
            torch::Tensor all_ones = torch::ones_like(result);
            torch::Tensor comparison = torch::eq(result, all_ones);
            bool all_equal = comparison.all().item<bool>();
            
            // This should always be true for a correctly functioning ones_like
            if (!all_equal) {
                throw std::runtime_error("ones_like did not produce all ones");
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