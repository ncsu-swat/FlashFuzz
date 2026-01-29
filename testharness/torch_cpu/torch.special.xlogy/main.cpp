#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for torch.special.xlogy
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // 1. Call xlogy with two tensors
        try {
            torch::Tensor result1 = torch::special::xlogy(x, y);
        } catch (...) {
            // Silently catch shape/dtype mismatches
        }
        
        // 2. Call xlogy with scalar and tensor
        if (Size > offset) {
            double scalar_value = static_cast<double>(Data[offset++]) / 10.0;
            try {
                torch::Tensor result2 = torch::special::xlogy(scalar_value, y);
            } catch (...) {
                // Silently catch errors
            }
        }
        
        // 3. Call xlogy with tensor and scalar
        if (Size > offset) {
            double scalar_value = static_cast<double>(Data[offset++]) / 10.0;
            try {
                torch::Tensor result3 = torch::special::xlogy(x, scalar_value);
            } catch (...) {
                // Silently catch errors
            }
        }
        
        // 4. Call xlogy with out parameter - need to handle broadcast shape
        try {
            // Compute expected output shape via broadcasting
            auto broadcasted = torch::broadcast_tensors({x, y});
            torch::Tensor out = torch::empty_like(broadcasted[0]);
            torch::special::xlogy_out(out, x, y);
        } catch (...) {
            // Silently catch errors
        }
        
        // 5. Try with different dtypes
        if (Size > offset + 2) {
            auto dtype_selector = Data[offset++] % 2;
            torch::ScalarType dtype;
            
            switch (dtype_selector) {
                case 0:
                    dtype = torch::kFloat;
                    break;
                case 1:
                    dtype = torch::kDouble;
                    break;
                default:
                    dtype = torch::kFloat;
            }
            
            try {
                torch::Tensor x_converted = x.to(dtype);
                torch::Tensor y_converted = y.to(dtype);
                torch::Tensor result_converted = torch::special::xlogy(x_converted, y_converted);
            } catch (...) {
                // Silently catch conversion/computation errors
            }
        }
        
        // 6. Try with broadcasting
        if (Size > offset && x.dim() > 0 && y.dim() > 0) {
            try {
                std::vector<int64_t> broadcast_shape;
                if (x.dim() > 1) {
                    broadcast_shape.push_back(x.size(0));
                    broadcast_shape.push_back(1);
                } else {
                    broadcast_shape.push_back(1);
                }
                
                torch::Tensor broadcast_tensor = torch::ones(broadcast_shape, x.options());
                torch::Tensor result_broadcast = torch::special::xlogy(x, broadcast_tensor);
            } catch (...) {
                // Silently catch broadcasting errors
            }
        }
        
        // 7. Test with edge cases - zeros and negative values
        if (Size > offset) {
            try {
                torch::Tensor zeros = torch::zeros_like(x);
                torch::Tensor result_zero_x = torch::special::xlogy(zeros, y);
            } catch (...) {
                // Silently catch errors
            }
            
            try {
                torch::Tensor zeros = torch::zeros_like(y);
                torch::Tensor result_zero_y = torch::special::xlogy(x, zeros);
            } catch (...) {
                // Silently catch errors - xlogy(x, 0) should return 0 when x=0, -inf otherwise
            }
        }
        
        // 8. Test with negative values (xlogy is defined for y > 0, but let's test edge cases)
        if (Size > offset) {
            try {
                torch::Tensor neg_y = torch::abs(y) + 0.1;  // Ensure positive for valid domain
                torch::Tensor result_valid = torch::special::xlogy(x, neg_y);
            } catch (...) {
                // Silently catch errors
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