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
        
        // Parse n parameter (optional)
        int64_t n = -1;  // Default: -1 means use default size
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim parameter
        int64_t dim = 0;  // Default to first dimension
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, make dim valid by taking modulo
            if (input.dim() > 0) {
                dim = ((dim % input.dim()) + input.dim()) % input.dim();
            }
        }
        
        // Parse norm parameter
        std::string norm = "backward";  // Default value
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 4) {
                case 0: norm = "backward"; break;
                case 1: norm = "forward"; break;
                case 2: norm = "ortho"; break;
                case 3: norm = ""; break;  // Empty string for None
            }
        }
        
        // Apply rfft operation
        torch::Tensor result;
        if (n == -1) {
            // Use default n
            result = torch::fft::rfft(input, c10::nullopt, dim, norm);
        } else {
            result = torch::fft::rfft(input, n, dim, norm);
        }
        
        // Perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
        // Try inverse operation to test round-trip
        if (result.numel() > 0) {
            auto inverse = torch::fft::irfft(result, n, dim, norm);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}