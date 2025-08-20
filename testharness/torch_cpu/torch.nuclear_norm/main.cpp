#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for nuclear_norm
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Get dim parameter if we have more data
        std::vector<int64_t> dim;
        if (offset < Size) {
            int64_t ndim = input.dim();
            if (ndim > 0) {
                uint8_t dim_selector = Data[offset++];
                int64_t dim_value = dim_selector % ndim;
                dim.push_back(dim_value);
                
                // Possibly add a second dimension for matrix-like operations
                if (offset < Size && ndim > 1) {
                    dim_selector = Data[offset++];
                    int64_t dim_value2 = dim_selector % ndim;
                    // Ensure second dimension is different from first
                    if (dim_value2 != dim_value) {
                        dim.push_back(dim_value2);
                    }
                }
            }
        }
        
        // Apply nuclear_norm operation
        torch::Tensor result;
        
        // Different ways to call nuclear_norm
        if (dim.empty()) {
            // Call without dim parameter
            result = torch::nuclear_norm(input, keepdim);
        } else {
            // Call with dim parameter
            result = torch::nuclear_norm(input, dim, keepdim);
        }
        
        // Try to access result to ensure computation is performed
        if (result.defined()) {
            auto item = result.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}