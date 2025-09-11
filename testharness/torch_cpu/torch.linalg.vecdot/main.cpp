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
        
        // Need at least two tensors for vecdot operation
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's data left
        if (offset < Size) {
            torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Get a dimension value for the dim parameter if there's data left
            int64_t dim = 0;
            if (offset < Size) {
                // Extract a dimension value from the remaining data
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else if (offset < Size) {
                    // Not enough data for full int64_t, use a single byte
                    dim = static_cast<int64_t>(Data[offset++]);
                }
            }
            
            // Apply the vecdot operation
            // vecdot computes the dot product of two batches of vectors along a specified dimension
            torch::Tensor result;
            
            // Try different variants of the vecdot operation
            if (offset < Size && Data[offset] % 2 == 0) {
                // Variant 1: With explicit dimension
                result = torch::vecdot(x, y, dim);
            } else {
                // Variant 2: Use default dimension (last dimension)
                result = torch::vecdot(x, y);
            }
            
            // Force computation to ensure any errors are triggered
            result.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
