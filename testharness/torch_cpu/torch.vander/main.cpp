#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - vander expects a 1-D tensor
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Flatten to 1D as vander requires 1-D input
        x = x.flatten();
        
        // Limit size to avoid memory issues (vander creates NxN matrix)
        if (x.numel() > 50) {
            x = x.slice(0, 0, 50);
        }
        
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
            // N should be positive and reasonable
            N = (std::abs(N) % 50) + 1;
        } else {
            N = x.numel(); // Default N is length of x
        }
        
        // Call torch::vander with different parameter combinations
        torch::Tensor result;
        
        // Try different combinations of parameters
        if (offset < Size) {
            uint8_t param_selector = Data[offset++] % 4;
            
            try {
                switch (param_selector) {
                    case 0:
                        // Just x - N defaults to len(x)
                        result = torch::vander(x);
                        break;
                    case 1:
                        // x and N
                        result = torch::vander(x, N);
                        break;
                    case 2:
                        // x and increasing (use x.numel() as N)
                        result = torch::vander(x, x.numel(), increasing);
                        break;
                    case 3:
                        // All parameters
                        result = torch::vander(x, N, increasing);
                        break;
                }
            } catch (const c10::Error&) {
                // Silently catch expected errors (e.g., invalid tensor types)
                return 0;
            }
        } else {
            // Default case with just x
            try {
                result = torch::vander(x);
            } catch (const c10::Error&) {
                return 0;
            }
        }
        
        // Basic validation - access elements to ensure computation completed
        if (result.defined() && result.numel() > 0) {
            auto sum = result.sum();
            (void)sum; // Prevent optimization
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}