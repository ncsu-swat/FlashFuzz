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
        
        // Need at least 2 bytes for basic parameters
        if (Size < 2) {
            return 0;
        }
        
        // Parse rank from the first byte
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse data type from the second byte
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Parse shape
        std::vector<int64_t> shape;
        if (offset < Size) {
            shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        }
        
        // Create zeros tensor with the parsed shape and dtype
        torch::Tensor zeros_tensor = torch::zeros(shape, torch::TensorOptions().dtype(dtype));
        
        // Test additional variants of zeros
        if (Size > offset && offset + 1 < Size) {
            // Test zeros_like
            torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor zeros_like_tensor = torch::zeros_like(input_tensor);
            
            // Test zeros with options
            bool requires_grad = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
            torch::Tensor zeros_with_options = torch::zeros(
                shape, 
                torch::TensorOptions()
                    .dtype(dtype)
                    .requires_grad(requires_grad)
            );
            
            // Test zeros with layout options
            if (offset < Size) {
                uint8_t layout_selector = Data[offset++];
                torch::Layout layout = (layout_selector % 2 == 0) ? torch::kStrided : torch::kSparse;
                
                torch::Tensor zeros_with_layout = torch::zeros(
                    shape,
                    torch::TensorOptions()
                        .dtype(dtype)
                        .layout(layout)
                );
            }
            
            // Test zeros with device options
            if (offset < Size) {
                torch::Tensor zeros_with_device = torch::zeros(
                    shape,
                    torch::TensorOptions()
                        .dtype(dtype)
                        .device(torch::kCPU)
                );
            }
            
            // Test zeros with memory format
            if (offset < Size) {
                uint8_t format_selector = Data[offset++];
                torch::MemoryFormat memory_format;
                
                switch (format_selector % 4) {
                    case 0: memory_format = torch::MemoryFormat::Contiguous; break;
                    case 1: memory_format = torch::MemoryFormat::Preserve; break;
                    case 2: memory_format = torch::MemoryFormat::ChannelsLast; break;
                    case 3: memory_format = torch::MemoryFormat::ChannelsLast3d; break;
                }
                
                // Only apply memory format if tensor has enough dimensions
                if ((memory_format != torch::MemoryFormat::ChannelsLast || shape.size() >= 3) &&
                    (memory_format != torch::MemoryFormat::ChannelsLast3d || shape.size() >= 5)) {
                    torch::Tensor zeros_with_memory_format = torch::zeros(
                        shape,
                        torch::TensorOptions()
                            .dtype(dtype)
                            .memory_format(memory_format)
                    );
                }
            }
        }
        
        // Verify that all elements are zero
        bool all_zeros = torch::all(zeros_tensor == 0).item<bool>();
        if (!all_zeros) {
            throw std::runtime_error("zeros tensor contains non-zero elements");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
