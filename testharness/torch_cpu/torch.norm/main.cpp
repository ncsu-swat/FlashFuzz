#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for norm operation if we have more data
        double p = 2.0; // Default p-norm
        int64_t dim = -1; // Default dimension
        bool keepdim = false;
        
        // Parse p value if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize p value - avoid NaN and extreme values
            if (std::isnan(p) || std::isinf(p)) {
                p = 2.0;
            }
            // Clamp p to reasonable range
            if (p > 1e6) p = 1e6;
            if (p < -1e6) p = -1e6;
        }
        
        // Parse dim value if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure dim is within valid range for the tensor
            if (input.dim() > 0) {
                dim = dim % input.dim();
                if (dim < 0) {
                    dim += input.dim();
                }
            } else {
                dim = 0;
            }
        }
        
        // Parse keepdim value if we have enough data
        if (offset < Size) {
            keepdim = Data[offset] & 0x1;
            offset++;
        }
        
        // Variant 1: Basic norm with default parameters
        try {
            torch::Tensor result1 = torch::norm(input);
            (void)result1;
        } catch (const std::exception&) {
            // May fail for certain tensor types
        }
        
        // Variant 2: Norm with specified p
        try {
            torch::Tensor result2 = torch::norm(input, p);
            (void)result2;
        } catch (const std::exception&) {
            // May fail for certain p values
        }
        
        // Variant 3: Norm with specified p and dim
        if (input.dim() > 0) {
            try {
                torch::Tensor result3 = torch::norm(input, p, {dim}, keepdim);
                (void)result3;
            } catch (const std::exception&) {
                // May fail for certain configurations
            }
        }
        
        // Variant 4: Frobenius norm (requires dimensions)
        if (input.dim() >= 2) {
            try {
                torch::Tensor result4 = torch::frobenius_norm(input, {-2, -1}, keepdim);
                (void)result4;
            } catch (const std::exception&) {
                // May require 2D tensor
            }
        } else if (input.dim() == 1) {
            try {
                torch::Tensor result4 = torch::frobenius_norm(input, {0}, keepdim);
                (void)result4;
            } catch (const std::exception&) {
            }
        }
        
        // Variant 5: Nuclear norm (requires 2D tensor)
        if (input.dim() == 2) {
            try {
                torch::Tensor result5 = torch::nuclear_norm(input);
                (void)result5;
            } catch (const std::exception&) {
                // Nuclear norm may not be supported for all tensor types
            }
        }
        
        // Variant 6: L1 norm
        try {
            torch::Tensor result6 = torch::norm(input, 1.0);
            (void)result6;
        } catch (const std::exception&) {
        }
        
        // Variant 7: Norm with infinity
        try {
            torch::Tensor result7 = torch::norm(input, INFINITY);
            (void)result7;
        } catch (const std::exception&) {
        }
        
        // Variant 8: Norm with negative infinity
        try {
            torch::Tensor result8 = torch::norm(input, -INFINITY);
            (void)result8;
        } catch (const std::exception&) {
        }
        
        // Variant 9: Norm with 0 (count of non-zero elements)
        try {
            torch::Tensor result9 = torch::norm(input, 0.0);
            (void)result9;
        } catch (const std::exception&) {
        }
        
        // Variant 10: Norm with negative p value
        if (p > 0 && p < 100) {
            try {
                torch::Tensor result10 = torch::norm(input, -p);
                (void)result10;
            } catch (const std::exception&) {
                // Negative p may not be supported
            }
        }
        
        // Variant 11: Norm with multiple dimensions
        if (input.dim() >= 2) {
            try {
                std::vector<int64_t> dims = {0, 1};
                torch::Tensor result11 = torch::norm(input, p, dims, keepdim);
                (void)result11;
            } catch (const std::exception&) {
            }
        }
        
        // Variant 12: Norm with fractional p value
        try {
            torch::Tensor result12 = torch::norm(input, 0.5);
            (void)result12;
        } catch (const std::exception&) {
        }
        
        // Variant 13: Frobenius norm with all dimensions
        if (input.dim() >= 3) {
            try {
                torch::Tensor result13 = torch::frobenius_norm(input, {-3, -2, -1}, keepdim);
                (void)result13;
            } catch (const std::exception&) {
            }
        }
        
        // Variant 14: Nuclear norm with keepdim
        if (input.dim() == 2) {
            try {
                torch::Tensor result14 = torch::nuclear_norm(input, keepdim);
                (void)result14;
            } catch (const std::exception&) {
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