#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// Helper function to check if two shapes are broadcastable
bool are_broadcastable(const torch::IntArrayRef& shape1, const torch::IntArrayRef& shape2) {
    size_t ndim1 = shape1.size();
    size_t ndim2 = shape2.size();
    size_t max_ndim = std::max(ndim1, ndim2);
    
    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
        int64_t dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    return true;
}

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create the first tensor (x) - must be floating point for ldexp
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to floating point if needed (ldexp requires float/double/complex)
        if (!x.is_floating_point() && !x.is_complex()) {
            x = x.to(torch::kFloat32);
        }
        
        // Create the exponent tensor (must be integer type)
        torch::Tensor exponent;
        if (offset < Size) {
            exponent = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to int32 as required by ldexp
            exponent = exponent.to(torch::kInt32);
            
            // Ensure exponent is broadcastable to x's shape
            // If shapes don't match, try to broadcast or use scalar
            if (!are_broadcastable(x.sizes(), exponent.sizes())) {
                // Shapes not broadcastable, use a scalar tensor instead
                int exp_val = exponent.numel() > 0 ? exponent.flatten()[0].item<int>() : 1;
                // Clamp to reasonable range to avoid overflow
                exp_val = std::max(-100, std::min(100, exp_val));
                exponent = torch::full(x.sizes(), exp_val, torch::kInt32);
            }
        } else {
            // Create a simple exponent tensor with same shape as x
            exponent = torch::ones(x.sizes(), torch::kInt32);
        }
        
        // Test 1: Basic in-place ldexp_
        {
            torch::Tensor x_copy = x.clone();
            try {
                x_copy.ldexp_(exponent);
            } catch (...) {
                // Shape mismatch or other expected failure
            }
        }
        
        // Test 2: Compare with out-of-place version
        {
            try {
                torch::Tensor result = torch::ldexp(x, exponent);
                (void)result;
            } catch (...) {
                // Expected failure for some inputs
            }
        }
        
        // Test 3: With scalar exponent
        if (offset + sizeof(int8_t) <= Size) {
            int8_t scalar_exp = static_cast<int8_t>(Data[offset++]);
            // Clamp to reasonable range
            scalar_exp = std::max(static_cast<int8_t>(-50), std::min(static_cast<int8_t>(50), scalar_exp));
            
            try {
                torch::Tensor x_copy = x.clone();
                torch::Tensor scalar_tensor = torch::full(x.sizes(), static_cast<int>(scalar_exp), torch::kInt32);
                x_copy.ldexp_(scalar_tensor);
            } catch (...) {
                // Expected failure
            }
        }
        
        // Test 4: With different floating point dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::ScalarType dtype;
            
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
                default: dtype = torch::kFloat32; break;
            }
            
            try {
                torch::Tensor x_typed = x.to(dtype);
                torch::Tensor x_typed_copy = x_typed.clone();
                // Exponent stays as int32
                x_typed_copy.ldexp_(exponent);
            } catch (...) {
                // Type conversion or operation failed
            }
        }
        
        // Test 5: With contiguous vs non-contiguous tensor
        if (x.dim() >= 2 && x.size(0) > 1 && x.size(1) > 1) {
            try {
                torch::Tensor x_transposed = x.transpose(0, 1).clone();
                torch::Tensor exp_transposed = exponent.dim() >= 2 ? 
                    exponent.transpose(0, 1).clone() : exponent.clone();
                x_transposed.ldexp_(exp_transposed);
            } catch (...) {
                // Expected failure for some shapes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}