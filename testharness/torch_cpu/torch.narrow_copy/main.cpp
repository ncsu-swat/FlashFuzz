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
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for narrow_copy from the remaining data
        if (offset + 3 >= Size) {
            return 0;
        }
        
        // Get dimension to narrow along
        int64_t dim = static_cast<int64_t>(Data[offset++]);
        if (input.dim() > 0) {
            dim = dim % input.dim();
        } else {
            // For 0-dim tensor, we'll just use 0 and let PyTorch handle the error
            dim = 0;
        }
        
        // Get start position
        int64_t start = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get length to narrow
        int64_t length = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply narrow_copy operation
        // We don't add defensive checks - let PyTorch handle invalid inputs
        torch::Tensor result = torch::narrow_copy(input, dim, start, length);
        
        // Basic sanity check to ensure the result is used
        if (result.defined()) {
            volatile auto num_elements = result.numel();
            (void)num_elements;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
