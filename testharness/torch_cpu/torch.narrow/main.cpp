#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for narrow operation
        // We need: dim, start, length
        
        // Get dim parameter (dimension along which to narrow)
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure dim is within valid range [-ndim, ndim-1]
            if (input_tensor.dim() > 0) {
                dim = dim % (2 * input_tensor.dim());
                if (dim < 0) {
                    dim += input_tensor.dim();
                } else if (dim >= input_tensor.dim()) {
                    dim -= input_tensor.dim();
                }
            } else {
                // For 0-dim tensors, any dim value is invalid, but we'll let PyTorch handle the error
                // No adjustment needed
            }
        }
        
        // Get start parameter (starting position)
        int64_t start = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // No adjustment for start - we want to test negative values too
        }
        
        // Get length parameter (length of slice)
        int64_t length = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // No adjustment for length - we want to test negative values too
        }
        
        // Apply narrow operation
        torch::Tensor result;
        result = torch::narrow(input_tensor, dim, start, length);
        
        // Verify result is not empty (basic sanity check)
        if (result.numel() > 0) {
            // Access some element to ensure computation is performed
            auto item = result.item();
        }
        
        // Alternative syntax test
        if (input_tensor.dim() > 0) {
            // Try the method version of narrow
            torch::Tensor result2 = input_tensor.narrow(dim, start, length);
            
            // Access some element to ensure computation is performed
            if (result2.numel() > 0) {
                auto item2 = result2.item();
            }
        }
        
        // Test with different dimensions if tensor has multiple dimensions
        if (input_tensor.dim() > 1) {
            int64_t alt_dim = (dim + 1) % input_tensor.dim();
            try {
                torch::Tensor result3 = torch::narrow(input_tensor, alt_dim, start, length);
                
                // Access some element to ensure computation is performed
                if (result3.numel() > 0) {
                    auto item3 = result3.item();
                }
            } catch (const std::exception &) {
                // Ignore exceptions for this alternative test
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}