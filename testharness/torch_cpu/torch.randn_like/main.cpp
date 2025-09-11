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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply randn_like operation
        torch::Tensor output_tensor = torch::randn_like(input_tensor);
        
        // Try with different options
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with different dtype
            if (option_byte & 0x01) {
                torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset % Size]);
                output_tensor = torch::randn_like(input_tensor, dtype);
            }
            
            // Try with different layout
            if (option_byte & 0x02) {
                torch::Layout layout = (Data[offset % Size] % 2 == 0) ? 
                    torch::kStrided : torch::kSparse;
                output_tensor = torch::randn_like(input_tensor, 
                    torch::TensorOptions().layout(layout));
            }
            
            // Try with different device
            if (option_byte & 0x04) {
                output_tensor = torch::randn_like(input_tensor, 
                    torch::TensorOptions().device(torch::kCPU));
            }
            
            // Try with different requires_grad
            if (option_byte & 0x08) {
                bool requires_grad = (Data[offset % Size] % 2 == 0);
                output_tensor = torch::randn_like(input_tensor, 
                    torch::TensorOptions().requires_grad(requires_grad));
            }
            
            // Try with memory format
            if (option_byte & 0x10) {
                torch::MemoryFormat memory_format;
                switch (Data[offset % Size] % 4) {
                    case 0: memory_format = torch::MemoryFormat::Contiguous; break;
                    case 1: memory_format = torch::MemoryFormat::Preserve; break;
                    case 2: memory_format = torch::MemoryFormat::ChannelsLast; break;
                    case 3: memory_format = torch::MemoryFormat::ChannelsLast3d; break;
                }
                output_tensor = torch::randn_like(input_tensor, 
                    torch::TensorOptions().memory_format(memory_format));
            }
            
            // Try with combined options
            if (option_byte & 0x20) {
                torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset % Size]);
                bool requires_grad = (Data[(offset + 1) % Size] % 2 == 0);
                output_tensor = torch::randn_like(input_tensor, 
                    torch::TensorOptions()
                        .dtype(dtype)
                        .requires_grad(requires_grad));
            }
        }
        
        // Verify the output tensor has the same shape as the input
        if (input_tensor.sizes() != output_tensor.sizes()) {
            throw std::runtime_error("Output tensor shape doesn't match input tensor shape");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
