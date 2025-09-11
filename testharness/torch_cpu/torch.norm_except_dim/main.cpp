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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for norm_except_dim
        // We need: pow, dim (note: keepdim is not supported)
        
        // Get pow parameter (norm type)
        int64_t pow = 2; // Default pow value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&pow, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get dim parameter (which dimension to exclude from norm calculation)
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply norm_except_dim operation
        torch::Tensor result = torch::norm_except_dim(input, pow, dim);
        
        // Optionally, perform some basic validation on the result
        if (result.numel() > 0) {
            // Access some elements to ensure computation completed
            auto accessor = result.accessor<float, 1>();
            volatile float first_element = accessor[0];
            (void)first_element; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
