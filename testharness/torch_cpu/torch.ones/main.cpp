#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Parse shape for the tensor
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Parse dtype from the next byte if available
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++];
            dtype = fuzzer_utils::parseDataType(dtype_byte);
        }
        
        // Create options with the parsed dtype
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Test torch::ones with different parameters
        try {
            // Basic ones with shape and options
            auto tensor1 = torch::ones(shape, options);
            
            // Test with just shape
            auto tensor2 = torch::ones(shape);
            
            // Test with shape and dtype directly
            auto tensor3 = torch::ones(shape, dtype);
            
            // Test with IntArrayRef directly if we have a non-empty shape
            if (!shape.empty()) {
                auto tensor4 = torch::ones({shape[0]});
            }
            
            // Test with a scalar if we have at least one dimension
            if (!shape.empty() && shape[0] > 0) {
                auto tensor5 = torch::ones(shape[0]);
            }
            
            // Test with memory format if available
            if (offset < Size) {
                uint8_t format_byte = Data[offset++];
                auto memory_format = format_byte % 4;
                
                switch (memory_format) {
                    case 0:
                        torch::ones(shape, options.memory_format(torch::MemoryFormat::Contiguous));
                        break;
                    case 1:
                        torch::ones(shape, options.memory_format(torch::MemoryFormat::Preserve));
                        break;
                    case 2:
                        torch::ones(shape, options.memory_format(torch::MemoryFormat::ChannelsLast));
                        break;
                    case 3:
                        torch::ones(shape, options.memory_format(torch::MemoryFormat::ChannelsLast3d));
                        break;
                }
            }
            
            // Test with layout if available
            if (offset < Size) {
                uint8_t layout_byte = Data[offset++];
                auto layout = layout_byte % 3;
                
                switch (layout) {
                    case 0:
                        torch::ones(shape, options.layout(torch::kStrided));
                        break;
                    case 1:
                        torch::ones(shape, options.layout(torch::kSparse));
                        break;
                    case 2:
                        torch::ones(shape, options.layout(torch::kSparseCsr));
                        break;
                }
            }
            
            // Test with device if available
            if (offset < Size) {
                uint8_t device_byte = Data[offset++];
                auto device_type = device_byte % 2;
                
                switch (device_type) {
                    case 0:
                        torch::ones(shape, options.device(torch::kCPU));
                        break;
                    case 1:
                        if (torch::cuda::is_available()) {
                            torch::ones(shape, options.device(torch::kCUDA));
                        }
                        break;
                }
            }
            
            // Test with requires_grad if available
            if (offset < Size) {
                uint8_t grad_byte = Data[offset++];
                bool requires_grad = grad_byte % 2 == 1;
                
                torch::ones(shape, options.requires_grad(requires_grad));
            }
            
            // Test with pin_memory if available
            if (offset < Size) {
                uint8_t pin_byte = Data[offset++];
                bool pin_memory = pin_byte % 2 == 1;
                
                torch::ones(shape, options.pinned_memory(pin_memory));
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and should be caught
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}