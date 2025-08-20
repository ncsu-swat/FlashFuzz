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
        
        // Parse offset parameter if we have more data
        int64_t offset_param = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&offset_param, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply diagflat operation
        torch::Tensor result;
        
        // Try different variants of diagflat
        if (offset < Size) {
            // Use the offset parameter if we have it
            result = torch::diagflat(input, offset_param);
        } else {
            // Default offset = 0
            result = torch::diagflat(input);
        }
        
        // Verify the result is a valid tensor
        if (!result.defined()) {
            throw std::runtime_error("diagflat returned undefined tensor");
        }
        
        // Try to access some properties to ensure the tensor is valid
        auto sizes = result.sizes();
        auto dtype = result.dtype();
        
        // Try to perform some operations on the result
        if (result.numel() > 0) {
            auto sum = torch::sum(result);
            auto mean = torch::mean(result);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}