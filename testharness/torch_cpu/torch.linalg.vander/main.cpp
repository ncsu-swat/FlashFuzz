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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse N parameter (number of columns in the output)
        int64_t N = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&N, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure N is within reasonable bounds
            N = std::abs(N) % 10 + 1; // 1 to 10
        } else {
            // Default value if not enough data
            N = 3;
        }
        
        // Parse increasing parameter
        bool increasing = false;
        if (offset < Size) {
            increasing = (Data[offset++] & 0x01) == 1;
        }
        
        // Apply torch.linalg.vander operation
        torch::Tensor result;
        
        // Try different variants of the operation
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            switch (variant) {
                case 0:
                    // Basic variant with default parameters
                    result = torch::vander(x);
                    break;
                    
                case 1:
                    // Variant with N specified
                    result = torch::vander(x, N);
                    break;
                    
                case 2:
                    // Variant with N and increasing specified
                    result = torch::vander(x, N, increasing);
                    break;
                    
                default:
                    // Fallback to basic variant
                    result = torch::vander(x);
                    break;
            }
        } else {
            // Default to basic variant if not enough data
            result = torch::vander(x);
        }
        
        // Perform a simple operation on the result to ensure it's used
        auto sum = result.sum();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
