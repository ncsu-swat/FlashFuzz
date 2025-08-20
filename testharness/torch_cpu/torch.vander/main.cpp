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
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for vander function if there's more data
        bool increasing = false;
        int64_t N = 0;
        
        if (offset + 1 < Size) {
            increasing = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&N, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Limit N to a reasonable range to avoid excessive memory usage
            N = std::abs(N) % 100;
        }
        
        // Call torch::vander with different parameter combinations
        torch::Tensor result;
        
        // Try different combinations of parameters
        if (offset < Size) {
            uint8_t param_selector = Data[offset++] % 4;
            
            switch (param_selector) {
                case 0:
                    // Just x
                    result = torch::vander(x);
                    break;
                case 1:
                    // x and N
                    result = torch::vander(x, N);
                    break;
                case 2:
                    // x and increasing
                    result = torch::vander(x, c10::nullopt, increasing);
                    break;
                case 3:
                    // All parameters
                    result = torch::vander(x, N, increasing);
                    break;
            }
        } else {
            // Default case with just x
            result = torch::vander(x);
        }
        
        // Basic validation - just access elements to ensure no segfaults
        if (result.defined() && result.numel() > 0) {
            auto first_element = result.flatten()[0];
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}