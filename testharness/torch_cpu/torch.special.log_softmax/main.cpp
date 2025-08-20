#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to apply log_softmax along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // If tensor has dimensions, use modulo to ensure dim is valid
        if (input.dim() > 0) {
            dim = dim % input.dim();
            if (dim < 0) {
                dim += input.dim();
            }
        }
        
        // Apply log_softmax operation
        torch::Tensor output = torch::special::log_softmax(input, dim, std::nullopt);
        
        // Try with optional dtype parameter if we have more data
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Apply log_softmax with dtype
            torch::Tensor output_with_dtype = torch::special::log_softmax(input, dim, dtype);
        }
        
        // Try with named dimension if tensor has named dimensions
        if (input.has_names() && input.dim() > 0) {
            auto dimname = input.names()[dim % input.dim()];
            torch::Tensor output_with_dimname = torch::special::log_softmax(input, dimname, std::nullopt);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}