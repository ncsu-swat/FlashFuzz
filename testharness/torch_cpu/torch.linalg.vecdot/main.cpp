#include "fuzzer_utils.h"
#include <iostream>
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
        
        // Need enough data for two tensors and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have data left for second tensor
        if (offset >= Size) {
            return 0;
        }
        
        // Create second input tensor with same shape as x for vecdot compatibility
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // vecdot requires tensors to have at least 1 dimension
        if (x.dim() == 0 || y.dim() == 0) {
            return 0;
        }
        
        // Make y have compatible shape with x by broadcasting/reshaping
        // vecdot requires the vectors along the specified dim to have same size
        try {
            // Ensure y has same shape as x for reliable testing
            y = y.expand_as(x).clone();
        } catch (...) {
            // If expansion fails, create y with same shape as x
            y = torch::randn_like(x);
        }
        
        // Get a dimension value for the dim parameter
        int64_t dim = -1; // default: last dimension
        if (offset < Size) {
            uint8_t dim_byte = Data[offset++];
            // Normalize dim to valid range
            dim = static_cast<int64_t>(dim_byte % x.dim());
            // Allow negative dims too
            if (offset < Size && Data[offset] % 2 == 0) {
                dim = dim - x.dim(); // Convert to negative indexing
            }
            if (offset < Size) offset++;
        }
        
        // Determine which variant to test
        uint8_t variant = (offset < Size) ? Data[offset++] % 3 : 0;
        
        torch::Tensor result;
        
        switch (variant) {
            case 0: {
                // Variant 1: With explicit dimension
                result = torch::linalg_vecdot(x, y, dim);
                break;
            }
            case 1: {
                // Variant 2: Use default dimension (last dimension, dim=-1)
                result = torch::linalg_vecdot(x, y);
                break;
            }
            case 2: {
                // Variant 3: Test with complex tensors
                try {
                    torch::Tensor x_complex = x.to(torch::kComplexFloat);
                    torch::Tensor y_complex = y.to(torch::kComplexFloat);
                    result = torch::linalg_vecdot(x_complex, y_complex, dim);
                } catch (...) {
                    // Fall back to real tensors if complex conversion fails
                    result = torch::linalg_vecdot(x, y, dim);
                }
                break;
            }
            default:
                result = torch::linalg_vecdot(x, y);
                break;
        }
        
        // Force computation to ensure any errors are triggered
        if (result.numel() > 0) {
            if (result.is_complex()) {
                result.abs().sum().item<float>();
            } else {
                result.sum().item<float>();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}